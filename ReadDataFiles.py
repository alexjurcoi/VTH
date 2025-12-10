import pandas as pd
import numpy as np
import matplotlib as mpl
import scipy as sc
import re
import os
import datetime

def readCV(filename: str, pH: float, area: float, referencePotential: float, compensationAmount: float = 0.15):
    """Reads cyclic voltammetry data from Biologic into Pandas dataframe.

    Args:
        filename (str): filename of CV .txt from biologic (must be exported using "CV-all") or .mpt file
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        compensationAmount (float, default 0.15): amount not compensated by potentiostat

    Returns:
        pd.DataFrame: dataframe of CV data
    """
    if filename[-3:] == 'mpt':
        
        #finds number of header lines and whether it's one of pedram's files
        with open(filename, 'r', encoding='windows-1252') as file:
            file.readline()
            line = file.readline()
            numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])
            
            headers = []
            
            #finds headers in file
            file.seek(0)
            currLine = 1
            for line in file:
                if currLine == numHeaderLines:
                    headers = line.strip().split('\t')
                currLine += 1

        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=numHeaderLines,
                        names = headers,
                        index_col=False,
                        dtype = np.float64,
                        encoding='windows-1252')
        
    else:
    
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=1,
                        names = ['mode',
                                    'ox/red',
                                    'error',
                                    'control changes',
                                    'counter inc.',
                                    'time/s',
                                    'control/V',
                                    'Ewe/V',
                                    'I/mA',
                                    'cycle number',
                                    '(Q-Qo)/C',
                                    'I Range',
                                    '<Ece>/V',
                                    'Analog IN 2/V',
                                    'Rcmp/Ohm',
                                    'P/W',
                                    'Ewe-Ece/V'],
                        index_col=False,
                        dtype = np.float64,
                        engine='python')
    
    if '<I>/mA' in data.columns:
        data['I/mA'] = data['<I>/mA']
    if '<Ewe>/V' in data.columns:
        data['Ewe/V'] = data['<Ewe>/V']
    if 'Rcmp/Ohm' in data.columns:
        solutionResistance = data['Rcmp/Ohm'].mean()
    else:
        if pH == 13:
            solutionResistance = 45
        else: #pH 14 usually
            solutionResistance = 5
    data['Ewe/V'] = data['Ewe/V'] - (data['I/mA']*0.001*solutionResistance*compensationAmount)
    data['I/A'] = data['I/mA']/1000
    data['j/mA*cm-2'] = data['I/mA']/area
    data['Ewe/V'] = data['Ewe/V'] + referencePotential + 0.059*pH
    data['Ewe/mV'] = data['Ewe/V']*1000
    data['j/A*m-2'] = data['j/mA*cm-2']*10
    
    #cleans data to remove points where Synology interferes with saving data
    data = data[~((data['time/s'] == 0) & (data['I/mA'] == 0))]
    
    return data

def buildCVList(filenameList: list[str], pH: float, area: float, referencePotential: float): 
    """Takes list of filenames with same pH, area, and referencePotential and builds CV list.

    Args:
        filenameList (list[str]): List of filenames to read CVs from.
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        list[pd.DataFrame]: List of DataFrames containing CV data.
    """
    dataList = []
    
    for filename in filenameList:
        data = readCV(filename,pH,area,referencePotential)
        dataList.append(data)
    
    return dataList

def buildEDLCList(folderName: str, number: int, pH: float, area: float, referencePotential: float, excludeLastX: int = 0):
    """Wrapper around buildCVList for faster EDLC analysis.

    Args:
        folderName (str): Folder that EDLC CVs are located in.
        number (int): Number that EDLC CVs all start with in common.
        pH (float): pH that EDLC was taken at.
        area (float): Area of electrode for EDLC in cm^2.
        referencePotential (float): Potential of reference electrode
        excludeLastX (int, optional): Excludes last x files in EDLC. Use this to remove faster scans in case they're less equilibrated. Defaults to 0.

    Returns:
        list[pd.DataFrame]: List of CVs for EDLC
    """
    twoDigit = False
    number = str(number)
    if len(number) == 2:
        twoDigit = True
    
    if not os.path.isdir(folderName):
        return None

    files = os.listdir(folderName)
    edlcFiles = []
    for file in files:
        isEDLC = False
        if twoDigit and (file[:2] == number):
            isEDLC = True
        elif (not twoDigit) and (file[0] == number):
            isEDLC = True
        if (file[-3:] != 'txt') and (file[-3:] != 'mpt'):
            isEDLC = False
        if ('CA' in file) or ('WAIT' in file):
            isEDLC = False
        if isEDLC:
            edlcFiles.append(folderName + '\\' + file)
    
    if excludeLastX != 0:        
        edlcFiles = edlcFiles[excludeLastX:]
        
    base_mpt = {f[:-4] for f in edlcFiles if f.endswith('.mpt')}
    cleaned_list = [f for f in edlcFiles if not (f.endswith('.txt') and f[:-4] in base_mpt)]

    return buildCVList(cleaned_list,pH,area,referencePotential)

def readCA(filename: str, pH: float, area: float, referencePotential: float, shouldRemoveNoise: bool = False, compensationAmount: float = 0.15): #area is in cm^2
    """Reads chronoamperometry data from Biologic into pandas dataframe.

    Args:
        filename (str): filename of CV .txt from biologic (must be exported using "CV-all") or .mpt file
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        shouldRemoveNoise (bool, default False): if true, will remove standard noise using ReadDataFiles.removeNoise
        compensationAmount (float, default 0.15): amount not compensated by potentiostat

    Returns:
        pd.DataFrame: dataframe of CA data
    """
    
    if filename[-3:] == 'mpt':
        
        #finds number of header lines
        with open(filename, 'r', encoding='windows-1252') as file:
            file.readline()
            line = file.readline()
            numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])
            
            #finds headers in file
            file.seek(0)
            currLine = 1
            for line in file:
                if currLine == numHeaderLines:
                    headers = line.strip().split('\t')
                currLine += 1

        data = pd.read_csv(filename,
                            sep='\s+',
                            skiprows=numHeaderLines,
                            names = headers,
                            index_col=False,
                            dtype = np.float64,
                            encoding='windows-1252')
            
    else:
    
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=1,
                        names = ['mode',
                                    'ox/red',
                                    'error',
                                    'control changes',
                                    'Ns changes',
                                    'counter inc.',
                                    'Ns',
                                    'time/s',
                                    'control/V',
                                    'Ewe/V',
                                    'I/mA',
                                    'dQ/C',
                                    'I range',
                                    'Ece/V',
                                    'Analog IN 2/V',
                                    'Rcmp/Ohm',
                                    'Capacitance charge/muF',
                                    'Capacitance discharge/muF',
                                    'Efficiency/%',
                                    'cycle number',
                                    'P/W'],
                        index_col=False,
                        dtype = np.float64,
                        encoding='unicode_escape')
    
    if '<I>/mA' in data.columns:
        data['I/mA'] = data['<I>/mA']
    if '<Ewe>/V' in data.columns:
        data['Ewe/V'] = data['<Ewe>/V']  
    if 'Rcmp/Ohm' in data.columns:
        solutionResistance = data['Rcmp/Ohm'].mean()
    else:
        if pH == 13:
            solutionResistance = 45
        else: #pH 14 usually
            solutionResistance = 5
    data['Ewe/V'] = data['Ewe/V'] - (data['I/mA']*0.001*solutionResistance*compensationAmount)
    
    if shouldRemoveNoise:
        data = removeNoise(data)

    data['j/mA*cm-2'] = data['I/mA']/area
    data['Ewe/V'] = data['Ewe/V'] + referencePotential + 0.059*pH
    data['Ewe/mV'] = data['Ewe/V']*1000
    data['j/A*m-2'] = data['j/mA*cm-2']*10
    
    #cleans data to remove points where Synology interferes with saving data
    data = data[~((data['time/s'] == 0) & (data['I/mA'] == 0))]
    
    return data

def buildCAList(filenameList: list[str], pH: float, area: float, referencePotential: float):
    """Takes list of filenames with same pH, area, and referencePotential and builds CA list.

    Args:
        filenameList (list[str]): List of filenames to read CVs from.
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        list[pd.DataFrame]: List of DataFrames containing CV data.
    """
    dataList = []
    
    for filename in filenameList:
        data = readCA(filename,pH,area,referencePotential)
        dataList.append(data)
    
    return dataList

def readPEIS(filename: str, freqRange: list[float] = None):
    """Reads potentiostatic electrochemical impedance spectroscopy data from Biologic into pandas dataframe.

    Args:
        filename (str): filename of PEIS .txt (must be exported using PEIS-all) or PEIS .mpt
        freqRange (list[float]): Frequency range to use for data, [low limit, high limit]. Defaults to None.

    Returns:
        pd.DataFrame: dataframe of PEIS data
    """
    if filename[-3:] == 'mpt':
        
        #finds number of header lines
        with open(filename, 'r', encoding='windows-1252') as file:
            file.readline()
            line = file.readline()
            numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])
            
            #finds headers in file
            file.seek(0)
            currLine = 1
            for line in file:
                if currLine == numHeaderLines:
                    headers = line.strip().split('\t')
                currLine += 1
            
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=numHeaderLines,
                        names = headers,
                        index_col=False,
                        dtype = np.float64,
                        encoding='windows-1252')
    else:
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=1,
                        names = ['freq/Hz',
                                    'Re(Z)/Ohm',
                                    '-Im(Z)/Ohm',
                                    '|Z|/Ohm',
                                    'Phase(Z)/deg',
                                    'time/s',
                                    '<Ewe>/V',
                                    '<I>/mA',
                                    'Cs/uF',
                                    'Cp/uF',
                                    'cycle number',
                                    'I Range',
                                    '|Ewe|/V',
                                    '|I|/A',
                                    'Ns',
                                    '(Q-Qo)/mA.h',
                                    'Re(Y)/Ohm-1',
                                    'Im(Y)/Ohm-1',
                                    '|Y|/Ohm-1',
                                    'Phase(Y)/deg',
                                    'dq/mA.h'],
                        index_col=False,
                        dtype = np.float64,
                        encoding='unicode_escape')
        
    #cleans data to remove points where Synology interferes with saving data
    data = data[~((data['time/s'] == 0) & (data['<I>/mA'] == 0))]
    
    #applies frequency range
    if freqRange != None:
        data = data[(data['freq/Hz'] >= freqRange[0]) & (data['freq/Hz'] <= freqRange[1])]
        
    data.attrs['filename'] = filename
        
    return data

def buildPEISList(filenameList: list[str], freqRange: list[float] = None):
    """Takes list of filenames and builds EIS list.

    Args:
        filenameList (list[str]): List of filenames to read EISs from.
        freqRange (list[float]): Frequency range to use for data, [low limit, high limit]. Defaults to None.

    Returns:
        list[pd.DataFrame]: List of DataFrames containing CV data.
    """
    dataList = []
    
    for filename in filenameList:
        data = readPEIS(filename, freqRange)
        dataList.append(data)
    
    return dataList

def readPEISImpedance(filename: str, freqRange: list[float] = None):
    """Reads PEIS directly into format that impedance.py or bayesdrt2 can use.

    Args:
        filename (str): filename of PEIS .txt (must be exported using PEIS-all) or PEIS .mpt
        freqRange (list[float]): [lowerFreq,upperFreq] defines frequency range to read file over. Defaults to None.

    Returns:
        tuple(np.ndarray(float),np.ndarray(complex)): (frequency, complex impedance)
    """
    return convertToImpedanceAnalysis(readPEIS(filename, freqRange))

def convertToImpedanceAnalysis(data: pd.DataFrame):
    """Converts dataframe to format that impedance.py or bayesdrt2 can use.

    Args:
        data (pd.DataFrame): Output of ReadDataFiles.readPEIS.

    Returns:
        tuple(np.ndarray(float),np.ndarray(complex)): (frequency, complex impedance)
    """
    
    frequency = data['freq/Hz'].to_numpy()
    dataLength = len(frequency)
    realImpedance = data['Re(Z)/Ohm'].to_numpy()
    imagImpedance = -data['-Im(Z)/Ohm'].to_numpy()
    impedance = np.zeros(dataLength,dtype=np.complex128)
    
    for i in range(0,dataLength):
        impedance[i] = complex(realImpedance[i],
                               imagImpedance[i])
    
    return (frequency,impedance)

def readOSC(filename: str,pH: float, area: float, referencePotential: float, irange: str, stretch: float = 1, solutionResistance: float = 0, compensationAmount: float = 1):
    """Reads oscilloscope .csv from Picoscope.

    Args:
        filename (str): filename of waveform .csv from Picoscope
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        irange (str): irange of measurement; '2A', '1A', '100mA', '10mA', '1mA' are acceptable
        stretch (float, optional): For 1 Hz, typically 4, for 10 Hz, typically 2. Defaults to 1.
        solutionResistance (float, default None): resistance to compensate for in Ohms. will try to find default using pH if not specified
        compensationAmount (float, default 1): amount not compensated by potentiostat

    Returns:
        pd.DataFrame: dataframe containing oscilloscope values
    """
    with open(filename,'r') as file:
        line1 = next(file)
        line2 = next(file)
        line3 = next(file)
        firstChannel = line1[33]
        unitsList = line2.split(',')
        unitsList = [i.replace('(','').replace(')','') for i in unitsList]
        timeUnit = unitsList[0]
        voltage1Unit = unitsList[3]
        voltage2Unit = unitsList[4]
        
    names = ['Time (xs)','discard1','discard2','ch1Avg','ch2Avg']
    
    data = pd.read_csv(filename,
                       skiprows=3,
                       engine='pyarrow',
                       sep=',',
                       names=names,
                       na_values=['∞'],
                       dtype=np.float32)
    
    data = data.ffill() #gets rid of na values
    
    #ensures correct units
    if 'mV' in voltage1Unit:
        data['ch1Avg'] = data['ch1Avg'] / 1000
    if 'mV' in voltage2Unit:
        data['ch2Avg'] = data['ch2Avg'] / 1000
    
    #ensures time starts at 0
    data['Time (xs)'] = data['Time (xs)'] - data['Time (xs)'].loc[0]
    
    #sets proper units of time
    if 'ms' in timeUnit:
        data['Time (ms)'] = data['Time (xs)']
    elif 'us' in timeUnit:
        data['Time (ms)'] = data['Time (xs)']/1000
        
    #ensures correct labels
    if 'A' in firstChannel:
        data['Voltage (V)'] = data['ch1Avg']
        data['Current (A)'] = data['ch2Avg']
    else:
        data['Voltage (V)'] = data['ch2Avg']
        data['Current (A)'] = data['ch1Avg']
    
    if irange == '1A' or irange == '2A':
        data['Current (A)'] = data['Current (A)']
    elif irange == '100mA':
        data['Current (A)'] = data['Current (A)']*0.1
    elif irange == '10mA':
        data['Current (A)'] = data['Current (A)']*0.01
    elif irange == '1mA':
        data['Current (A)'] = data['Current (A)']*0.001
        
    #compensates resistance (for some reason A seems to be much higher than V)
    #data['Voltage (V)'] = data['Voltage (V)'] - (data['Current (A)']*solutionResistance*compensationAmount)
    
    #does stretch if necessary
    data['Time (ms)'] *= stretch
    data = data[data['Time (ms)'] < (data['Time (ms)'].max()/stretch)]
    
    #converts to seconds
    data['Time (s)'] = data['Time (ms)']/1000
    
    #does RHE correction
    data['RawVoltage (V)'] = data['Voltage (V)']
    data['Voltage (V)'] = data['Voltage (V)'] + referencePotential + 0.059*pH
    
    #gets charge and charge density
    chargeArray = sc.integrate.cumulative_trapezoid(data['Current (A)'],
                                                    x = data['Time (s)'])
    chargeArray = np.insert(chargeArray,0,0)
    data['Charge (C)'] = pd.Series(chargeArray)
    data['Charge Density (mC/cm^2)'] = data['Charge (C)']*1000/area
    
    #gets current density
    data['Current (mA)'] = data['Current (A)']*1000
    data['Current Density (mA/cm^2)'] = data['Current (mA)']/area
    
    #drops unnecessary columns to save memory
    data = data.drop(['discard1','discard2','Time (xs)','ch1Avg','ch2Avg'],axis=1)

    return data

def readRawWaveform(filename: str, pH: float, referencePotential: float):
    """Reads waveforms that Siglent can read into a Pandas DataFrame.

    Args:
        filename (str): filename of waveform .csv from makeWaveform.py or from Siglent software.
        pH (float): pH of electrolyte for RHE conversion
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        pd.DataFrame: dataframe with 'Time (s)', 'Time (ms)', 'RawVoltage (V)', 'Voltage (V)'
    """
    with open(filename,'r') as file:
        lines = file.readlines()
        dataLength = float(lines[0][12:])
        frequency = float(lines[1][10:])
    
    data = pd.read_csv(filename,
                       skiprows=12,
                       sep=',',
                       dtype=float)
    data['Time (s)'] = (data['xpos']-1)/(frequency*dataLength)
    data['Time (ms)'] = data['Time (s)']*1000
    data['RawVoltage (V)'] = data['value']
    data['Voltage (V)'] = data['value'] + referencePotential + 0.059*pH
    
    return data

def readDRT(filename: str):
    """Reads DRT generated by pyDRTTools.

    Args:
        filename (str): filename of DRT .csv

    Returns:
        pd.DataFrame: dataframe with 'tau', 'gamma'
    """
    data = pd.read_csv(filename,
                       skiprows=2,
                       sep=',',
                       dtype=float)
    return data

def colorFader(c1: str,c2: str,currentIndex: int, totalIndices: int):
    """Generates color gradient for plotting.

    Args:
        c1 (str): color of first index
        c2 (str): color of last index
        currentIndex (int): index to linearly interpolate between colors (1 is c1, totalIndices is c2)
        totalIndices (int): total number of indices

    Returns:
        color: matplotlib color in hex format
    """
    if totalIndices > 1:
        mix = currentIndex/(totalIndices-1)
    else:
        mix = 0
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def removeNoise(df: pd.DataFrame, fundamental_freq: float = 0.46, Q: float =1):
    """
    Apply multiple notch filters to remove a fundamental frequency and its harmonics
    from an electrochemical signal.
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'time/s' and 'I/mA' columns
    fundamental_freq : float, optional
        The fundamental frequency to remove in Hz, default is 0.46 Hz
    Q : float, optional
        Quality factor for the notch filters. Higher values create narrower notches.
        Default is 30.
        
    Returns:
    -------
    pandas.DataFrame
        A copy of the input DataFrame with filtered 'I/mA' values
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    filtered_df = df.copy()
    
    # Extract the time and current data
    time = df['time/s'].values
    current = df['I/mA'].values
    
    # Calculate sampling frequency from the time data
    # Using median difference to handle potential irregularities in sampling
    time_diffs = np.diff(time)
    dt = np.median(time_diffs)
    fs = 1/dt
    
    # Apply notch filter for fundamental and each harmonic
    filtered_current = current.copy()
    
    frequencies_removed = []
    
    belowNyquistFreq = True
    i = 1
    while belowNyquistFreq:
        target_freq = i * fundamental_freq
        
        # Skip if the frequency is above Nyquist frequency
        if target_freq > fs/2:
            belowNyquistFreq = False
            break
        
        # Create and apply the notch filter
        b, a = sc.signal.iirnotch(target_freq, Q, fs)
        filtered_current = sc.signal.filtfilt(b, a, filtered_current)
        
        frequencies_removed.append(target_freq)
        i += 1
    
    # Update the DataFrame with the filtered data
    filtered_df['I/mA'] = filtered_current
    
    return filtered_df

def calculateIntegral(timeSeries: pd.Series, valueSeries: pd.Series, baseline: float, timeBounds: list[float]):
    """Calculates 1D integral of (valueSeries-baseline) over timeSeries bounded by timeBounds.

    Args:
        timeSeries (pd.Series): x-values of integration
        valueSeries (pd.Series): y-values of integration
        baseline (float): amount to subtract valueSeries by
        timeBounds (list[float]): [0] is lower bound, [1] is upper bound

    Returns:
        float: value of integral
    """
    #truncates timeSeries and valueSeries according to timeBounds
    timeSeries = timeSeries[(timeSeries >= timeBounds[0]) & (timeSeries <= timeBounds[1])]
    valueSeries = valueSeries.loc[timeSeries.first_valid_index():timeSeries.last_valid_index()]
    
    #subtracts baseline from valueSeries
    valueSeries = valueSeries - baseline
    
    #integrates
    return sc.integrate.trapezoid(y=valueSeries,x=timeSeries)

def readExcelSheet(filename: str, area: float):
    """Gets H2 produced from Excel sheet. Ensure that the Excel sheet contains the correct analysis
    and that the Excel sheet names start with the experiment number.

    Args:
        filename (str): .xlsx file with GC data
        area (float): area of electrode in cm^2 for normalization
        
    Returns:
        dict[experimentNumber] = (H2 Value (mol/cm^2), H2 Error (mol/cm^2)): {int,(np.float64,np.float64)}
    """
    
    finalDict = {}
    
    xl = pd.ExcelFile(filename)

    for sheet_name in xl.sheet_names:
        
        try:
            experimentNumber = int(re.search(r'\d+', str(sheet_name)).group())
        except:
            continue
        
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        
        # Search for the cell in this sheet
        foundValue = False
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                if isinstance(value, str) and "Total H2 Generated (mol)" in value:
                    h2prod = np.float64(df.iloc[row_idx, col_idx + 1])
                    h2err = np.float64(df.iloc[row_idx, col_idx + 2])
                    foundValue = True
                    break
            if foundValue:
                break
            
        finalDict[experimentNumber] = (h2prod/area,h2err/area)
    
    return finalDict

def readOCV(filename: str, pH: float, referencePotential: float):
    """Reads open circuit potential .mpt file from Biologic.

    Args:
        filename (str): Filename of .mpt file from Biologic OCV experiment.
        pH (float): pH of electrolyte for RHE conversion
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        pd.DataFrame: Contains 'mode', 'error', 'Ewe/V', 'Ece/V', 'Analog IN 2/V', 'Ewe/mV'
    """
    #finds number of header lines
    with open(filename, 'r', encoding='windows-1252') as file:
        file.readline()
        line = file.readline()
        numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])
        
    data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=numHeaderLines,
                        names = ['mode',
                                 'error',
                                 'time/s',
                                 'Ewe/V',
                                 'Ece/V',
                                 'Analog IN 2/V',
                                 'Ewe-Ece/V'],
                        index_col=False,
                        dtype = np.float64,
                        encoding='windows-1252')
    
    data['Ewe/V'] = data['Ewe/V'] + referencePotential + 0.059*pH
    data['Ece/V'] = data['Ece/V'] + referencePotential + 0.059*pH
    data['Ewe/mV'] = data['Ewe/V']*1000
    
    
    return data
readOCV('1.5UC-SRO-BTO-SRO-NSTO(3rd_sample_new)-1MKOH_GraphiteCE_SCE_REF_CV_upvsdown_04_CV_C01.mpt', 14, 0.241)
def buildTechniqueList(folder_path: str, techniqueName: str):
    """
    Finds all files of a specific technique from .mpt files in a folder
    and returns them in chronological order (earliest acquisition first). 
    Mostly if files are mislabeled.
    
    Args:
        folder_path (str): Path to the folder to search in.
        techniqueName (str): Technique name. Common are 'Cyclic Voltammetry', 
                             'Potentio Electrochemical Impedance Spectroscopy', 
                             'Chronoamperometry / Chronocoulometry'
        
    Returns:
        list[str]: List of full file paths to this technique in the folder, sorted by acquisition time
    """
    
    matching_files = []
    search_string = techniqueName
    
    # Verify the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' does not exist.")
        return matching_files
    
    # Get all .mpt files in the folder
    mpt_files = [f for f in os.listdir(folder_path) if f.endswith('.mpt')]
    
    if not mpt_files:
        print(f"No .mpt files found in '{folder_path}'.")
        return matching_files
    
    # Check each file
    file_info = []  # Will store tuples of (file_path, acquisition_datetime)
    for filename in mpt_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='windows-1252') as file:
                # Check technique on fourth line
                lines = file.readlines(1000)
                #print('hi')
                #print(len(lines))
                if len(lines) >= 4:
                    fourth_line = lines[3]
                    secondLine = lines[1]
                    if search_string in fourth_line:
                        # Look for acquisition date/time line
                        acquisition_time = None
                        for line in lines:
                            if "Acquisition started on :" in line:
                                # Extract the date/time using regex
                                date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}\.\d{3})', line)
                                if date_match:
                                    date_str = date_match.group(1)
                                    try:
                                        # Parse the date/time string
                                        acquisition_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f')
                                        break
                                    except ValueError:
                                        # If parsing fails, try alternative format
                                        try:
                                            acquisition_time = datetime.datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S.%f')
                                            break
                                        except ValueError:
                                            pass
                        
                        # If we found a valid acquisition time, add to list
                        if acquisition_time:
                            file_info.append((file_path, acquisition_time))
                        else:
                            # Fallback to modification time if acquisition time not found
                            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                            file_info.append((file_path, mod_time))
                    elif 'Nb header lines : 3' in secondLine: #check if it's EC-Lab Express file meant to be read as a CA
                        if search_string == 'Chronoamperometry / Chronocoulometry':
                            # Fallback to modification time if acquisition time not found (no acquisition time in these EC-Lab Express Dynamic CAs)
                            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                            file_info.append((file_path, mod_time))
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
    
    # Sort the matched files by acquisition time (earliest first)
    file_info.sort(key=lambda x: x[1])
    
    # Extract just the file paths from the sorted list
    matching_files = [info[0] for info in file_info]
    
    #prints matching_files
    for i, file in enumerate(matching_files):
        print(str(i)+' '+file)
        print('\n')
    
    return matching_files

def split_by_known_prefixes(file_paths, known_prefixes):
  """
  Split a list of file paths based on known filename prefixes.
  
  Args:
      file_paths (list): List of full file paths
      known_prefixes (list): List of known prefixes to check for
      
  Returns:
      dict: Dictionary where keys are prefixes and values are lists of full file paths,
            with an additional "unmatched" key for files that don't match any prefix
  """
  # Initialize the result dictionary with empty lists for each prefix
  result = {prefix: [] for prefix in known_prefixes}
  result["unmatched"] = []  # Add the unmatched category
  
  for path in file_paths:
      # Extract just the filename
      filename = os.path.basename(path)
      
      # Check which known prefix this filename starts with
      matched = False
      for prefix in known_prefixes:
          if filename.startswith(prefix):
              result[prefix].append(path)
              matched = True
              break
      
      # If no prefix matched, add to the unmatched list
      if not matched:
          result["unmatched"].append(path)
  
  return result