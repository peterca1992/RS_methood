import yfinance as yf
import numpy as np


df = yf.download("2330.TW", start = "2010-01-01", end = "2025-12-31", auto_adjust = True)

df.columns = ["close", "high", "low", "open", "volume"]
df = df.reset_index(drop = False)


upbound_list = [np.NaN, np.NaN]
lowbound_list = [np.NaN, np.NaN]
sign_list = [np.NaN, np.NaN]

for i in range(2,len(df)):
    
    #進場（三天不過高 建立高點)
    if(df["high"].iloc[i-2] == max(df["high"].iloc[(i-2) : (i+1)])):
        
        upbound_point = df["high"].iloc[i-2] #產生高點
        lowbound_point = upbound_point * 0.9
        
        #判定新的upbpund 是否取代舊的 並且判定是否出場
        if(np.isnan(upbound_list[i-1]) == False and upbound_point >= lowbound_list[i-1]):
            
            upbound_point = max(upbound_list[i-1], upbound_point) 
            lowbound_point = upbound_point * 0.9
            
            upbound_list.append(upbound_point)
            lowbound_list.append(lowbound_point)
            sign_list.append("續抱")
            
        else:
            upbound_list.append(upbound_point)
            lowbound_list.append(lowbound_point)
            sign_list.append("進場")
    
    #沒進場
    else:
        
        #已進場
        if(sign_list[i-1] == "續抱" or sign_list[i-1] == "進場"):
            
            #判斷是否出場
            if(df["close"].iloc[i] > upbound_list[i-1] or df["close"].iloc[i] < lowbound_list[i-1]):
                
                upbound_list.append(np.NaN)
                lowbound_list.append(np.NaN)
                sign_list.append("出場")
                    
            
            else:
                
                upbound_list.append(upbound_list[i-1])
                lowbound_list.append(lowbound_list[i-1])
                sign_list.append(sign_list[i-1])
        
        #未已進場
        else:
            
            upbound_list.append(upbound_list[i-1])
            lowbound_list.append(lowbound_list[i-1])
            sign_list.append(sign_list[i-1])


df["upbound"] = upbound_list
df["lowbound"] = lowbound_list
df["sign"] = sign_list













