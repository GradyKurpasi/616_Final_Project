
import os
from pandas.io.parsers import ParserBase
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd 
import numpy as np 
import torch
import argparse

# from torch.utils.data.dataset import BufferedShuffleDataset


arg_data_path = "./data/"   #default to local environment
arg_num_epochs = 100         #default value for local run

def ParseArguments():
    """
        AZUREML SDK CODE 
        Parse Arguments from ScripRun
        DO NOT CALL FOR LOCAL RUNS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest="data_folder", help="data folder mounting point", default="")
    parser.add_argument("--num-epochs", type=int, dest="num_epochs", help="Number of epochs", default="")
    args = parser.parse_args()
    global arg_data_path
    arg_data_path = args.data_folder
    global arg_num_epochs
    arg_num_epochs = args.num_epochs

def LoadData(datapath = None, filename = None):
    """ 
        Loads data from Master Output.csv and Payment Output.csv, sets datatypes 
        Accepts optional filename parameter
        filename MUST be a formated as a 'master' output file
    """
    if datapath == None: datapath = arg_data_path
    if filename == None: filename = 'Master_Output_Short.csv'
        # filename = 'Master_Output_Clean.csv'
        # filename = 'Master_Output_Trunc.csv'
    filename = os.path.join(datapath, filename)
    master = pd.read_csv(filename, dtype={
        'clientname' : str,
        'accountid' : int,
        'customerid' : int,
        'ssnlastfour' : int,
        'Address1' : str,
        'Address2' : str,
        'City' : str,
        'State' : str,
        'PostalCode' : str,
        'dateplaced' : str,
        'placedbalance' : float,
        'dollarscollected' : float,
        'hasbrokenpromise' : int,
        'hasconfirmed' : int,
        'haskeptptp' : int,
        'haspayments' : int,
        'hasplacementphone' : int,
        'haspossiblenumber' : int,
        'haspromever' : int
    }) 

    filename = os.path.join(datapath, 'Payment_Output.csv')
    payments =  pd.read_csv(filename, dtype ={
        'accountid' : int,
        'customerid' : int, 
        'Collected' : float,
        'IsNSF' : int,
        'IsPayment' : int,
        'MonthsWorked' : int,
        'DatePosted' : str        
    })
    
    return master, payments

def PrepMLPData(master, payments):
    """ 
        Adds Month 1 and Month 2 columns to master to reflect account performance over 60 days
        MLP prediction only looks at first 60 days of account performance and tries to predict the rest

        walks through master and payments simultaneously to construct solution string
        i.e. 0 or 1 if payment expected
        automatically 0 if no payment
        0 if no payment after 2 months
        1 if payments after 2 months

         
    """

    # convert date columns from string to datetime
    master['dateplaced'] = pd.to_datetime(master['dateplaced'])
    payments['DatePosted']= pd.to_datetime(payments['DatePosted'])
    
    #sort both dataframes
    master = master.sort_values(by=['customerid', 'accountid'])
    payments = payments.sort_values(by=['customerid', 'accountid', 'DatePosted'])
    
    # create ndarray placeholders
    month1 = np.zeros(len(master))
    month2 = np.zeros(len(master))
    solution = np.zeros(len(master))

    #Testing Code
    record = 0

    # walk through master
    for index, row in master.iterrows():
        
        #Testing code
        # if record % 1000 == 0: print(record)
        # if record >= 10000: break
        # record += 1

        #get current account data    
        account_id = row['accountid']
        customer_id = row['customerid']
        
        #get payment rows
        account_payments = payments.loc[(payments['customerid'] == customer_id) & (payments['accountid']==account_id)]
        if len(account_payments) == 0: continue 

        #walk through payment rows
        for pay_index, pay_row in account_payments.iterrows():

            #get months between payment and account placement
            month = (pay_row['DatePosted']-row['dateplaced']) // np.timedelta64(1, 'M') 

            if month == 1: # a payment was made in Month 1 
                month1[index] = 1
            elif month == 2:  # a payment was made in Month 2
                month2[index] = 1
            elif month >= 3: # there was future performance on account
                solution[index] = 1
                break
        
    master['Month1'] = month1
    master['Month2'] = month2
    
    return solution, master

    
def CreateMLPDataLoader(master, solution):
    """ 
        Converts dataframe -> ndarray -> tensor -> torch.DataSet -> torch.DataLoader
        Also splits data into training and testing data sets
    """

    #convert data frame to ndarray and split data into training and test datasets
    # master is a dataframe that needs to be converted
    # solution is already an ndarray, just needs to be split
    split = (len(master) // 3) * 2 # set split point at 2/3 of data
    train_arrays = master[:split].to_numpy(dtype='float64')
    test_arrays = master[split:].to_numpy(dtype='float64')
    train_solution = solution[:split]
    test_solution = solution[split:]

    #convert ndarray to tensor
    train_tensor = torch.tensor(np.stack(train_arrays))
    test_tensor = torch.tensor(np.stack(test_arrays))
    train_solution_tensor = torch.tensor(train_solution)
    test_solution_tensor = torch.tensor(test_solution)
    # data frame to tensor conversion could also have been done like this
    # list_of_tensors = [to rch.tensor(np.array(df)) for df in data]
    # torch.stack(list_of_tensors)

    #convert tensor to DataSet
    train_ds = TensorDataset(train_tensor, train_solution_tensor)
    test_ds = TensorDataset(test_tensor, test_solution_tensor)
    #convert DataSet to DataLoader
    train_dl = DataLoader(train_ds, batch_size=100, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=100, shuffle=True)

    return train_dl, test_dl


def PreProcess(datapath = None, trainfile = 'train_dl2.pt', testfile = 'test_dl2.pt', location = 'AZURE'):
    """
        Run All Preprocessing 
        NOT RECOMMENDED FOR LOCAL RUNS
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if location == 'AZURE': ParseArguments()
    if datapath == None: datapath = arg_data_path

    master, payments = LoadData(filename='alldone.csv')
    print('Data Loaded')
    solution, master = PrepMLPData(master, payments)
    print('Data Prepped')
    train_dl, test_dl = CreateMLPDataLoader(master, solution)
    print('Data Loader Created')
    filename = os.path.join(datapath, trainfile)
    torch.save(train_dl, filename)
    filename = os.path.join(datapath, testfile)
    torch.save(test_dl, filename)
    print('YAY!')

def LoadPreProcess(datapath = None, trainfile = 'train_dl.pt', testfile = 'test_dl.pt'):
    

    ParseArguments() # REMOVE FOR LOCAL RUNS
    if datapath == None: datapath = arg_data_path
    filename = os.path.join(datapath, trainfile)
    train_dl = torch.load(filename)
    filename = os.path.join(datapath, testfile)
    test_dl = torch.load(filename)
    return train_dl, test_dl



# UNCOMMENT FOR AZURE RUNS
# PreProcess()
# ParseArguments()
# train_dl, test_dl = LoadPreProcess()
# n = 0
# for x, y in train_dl:
#     print(x, y)
#     n += 1
#     if n > 5: break









##############################################################################################################################
# OLD COLD 
# DO NOT USE 
# MAKES MULTIPLE PASS THROUGH ALL ROWS IN DATA FRAME  
# VERY SLOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    # # convert all strings to integers
    # def strtoint(columnname):
    #     nonlocal master
    #     uniq = master[columnname].unique()
    #     for index, val in np.ndenumerate(uniq):
    #         master[columnname] = master[columnname].replace([val], index[0])

    # strtoint('clientname')
    # strtoint('Address1')  #THIS NEEDS TO BE REPLACED - PRETTY MUCH EVERY ADDRESS IS UNIQUE - VERRRRRRYYYY SLOOOOWWWWWW
    # strtoint('Address2')
    # strtoint('City')
    # strtoint('State')
    # strtoint('PostalCode')
    # strtoint('dateplaced')