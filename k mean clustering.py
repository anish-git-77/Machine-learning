import pandas as pd
import math
df=pd.read_csv(r'D:/KIDS_SCREEN_TIME2.CSV')
print(df)
x1=df["Urban_or_Rural"].to_list()
x2=df["Poor Sleep"].to_list()
x3=df["Eye Strain"].to_list()
x4=df["Obesity Risk"].to_list()
A1=1
A2=1
A3=1
A4=1
B1=1
B2=1
B3=0
B4=0
C1=1
C2=1
C3=0
C4=0
d1=[]
d2=[]
d3=[]
d4=[]
for i in range(2,245):
    d11=x1[i]-A1
    d111=x2[i]-A2
    d1111=x3[i]-A3 
    d11111=x4[i]-A4 
    dx=(d11*d11)+(d111*d111)+(d1111*d1111)+(d11111*d11111)
    dxy=math.sqrt(dx)
    d1.append(dxy)
#print(d1)    
for i in range(2,245):
    d22=x1[i]-B1
    d222=x2[i]-B2
    d2222=x3[i]-B3
    d22222=x4[i]-B4
    dx=(d22*d22)+(d222*d222)+(d2222*d2222)+(d22222*d22222)
    dxy=math.sqrt(dx)
    d2.append(dxy)
#print(d2)
for i in range(2,245):
    d33=x1[i]-C1
    d333=x2[i]-C2
    d3333=x3[i]-C3
    d33333=x4[i]-C4
    dx=(d33*d33)+(d333*d333)+(d3333*d3333)+(d33333*d33333)
    dxy=math.sqrt(dx)
    d3.append(dxy)
#print(d3
cluster=[]
cluster1=[]
cluster2=[]
cluster3=[]
for i in range(0,240):
    k1=min(d1[i],d2[i],d3[i])
    if(k1==d1[i]):
        cluster.append(1)
    if(k1==d2[i]):
        cluster.append(2)
    if(k1==d3[i]):
        cluster.append(3)
print(cluster)
for i in range(0,240):
    if(cluster[i]==1):
        cluster1.append(cluster[i])
    if(cluster[i]==2):
        cluster2.append(cluster[i])
    if(cluster[i]==3):
        cluster3.append(cluster[i])      
print("\nData of cluster 1")        
print(cluster1)
print("\nData of cluster 2")
print(cluster2)
print("\nData of cluster 3")
print(cluster3)
