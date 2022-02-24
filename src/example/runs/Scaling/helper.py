import numpy as np

def readHostData(Nsp, Nthreads, output):
    # 0. Source term, 1. analytic jac, 2. num jac fwd, 3. sacado jac
    data = np.zeros([4, len(Nsp), len(Nthreads)])
    for i,Nt in enumerate(Nthreads):
        for j,N in enumerate(Nsp):
            file_name = output+"/wall_times_"+str(Nt)+"_"+str(N)+".dat"
            temp_data = np.genfromtxt(file_name,
                     dtype={'names': ('computation type', 'total time', 'time per sample'), 'formats': ('S30', 'f16', 'f16')},
                     delimiter=",",
                     skip_header=1)
            for k in range(4):
                data[k,j,i] = temp_data[k][2]
    return data

def readDeviceData(Nsp,vector, team, output_times_2):
    format_line={'names': ('computation type', 'total time', 'time per sample'), 'formats': ('S30', 'f16', 'f16')}
    data = np.zeros([len(Nsp), len(team)])
    for m,N in enumerate(Nsp):
        for i in range(len(team)):
            last_name = "/Times_nBacth"+str(N)+"_V"+str(vector[i])+"T"+str(team[i])+".dat"
            output_2=output_times_2 + last_name
            temp_data2 = np.genfromtxt(output_2, dtype=format_line, delimiter=",",comments="#")
            try:
                data[m,i] = temp_data2[1][2]
            except : 
                print('File is empty')
                data[m,i] = np.nan
            else:
                data[m,i] = temp_data2[1][2]         
    return data

def makeTeamVectorLabel(vector,team):
    team_vector = {}
    for i in range(len(team)):
        team_vector.update({i:str(team[i])+"x"+str(vector[i])})
    return team_vector
