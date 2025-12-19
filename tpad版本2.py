import pandas as pd
import sys
sys.path.append('C:\\Users\\J1070116\\Desktop')
from WCFAdox import PCAX
import numpy as np
import time
from sklearn.linear_model import LinearRegression

#設定連線主機IP並產生物件
PX=PCAX("172.24.26.40")


#%%
#日資料更新
df_price = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\price.csv")
df_delta = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\delta.csv")
df_ret = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\ret.csv")
df_vol = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\vol.csv")

df_price["日期"] = pd.to_datetime(df_price["日期"], format = "%Y-%m-%d")
df_delta["日期"] = pd.to_datetime(df_delta["日期"], format = "%Y-%m-%d")
df_ret["日期"] = pd.to_datetime(df_ret["日期"], format = "%Y-%m-%d")
df_vol["日期"] = pd.to_datetime(df_vol["日期"], format = "%Y-%m-%d")

yy = str(df_price["日期"].iloc[-1].year)

mm = str(df_price["日期"].iloc[-1].month)
if(len(mm) == 1):
    mm = "0" + mm
    
dd = str(df_price["日期"].iloc[-1].day)
if(len(dd) == 1):
    dd = "0" + dd
    
start_day = yy + mm +dd

df = "select 日期, 股票代號, 股票名稱, 收盤價, 漲跌, [成交金額(千)], [總市值(億)] from  [dbo].[日收盤表排行] where (日期 > '" + start_day + "') and (len(股票代號) = 4)"
sqltables = "日收盤表排行"
df = PX.Sql_data(df, sqltables)

df = df[df.apply(lambda x : len(x["股票代號"]), axis = 1) == 4] #留下4碼的股票代號
df = df[df["股票代號"] >= "1101"].reset_index(drop = True)

df["日期"] = pd.to_datetime(df["日期"], format = "%Y%m%d")

#price
df_price_part = df[df["收盤價"] != ""].reset_index(drop = True)
df_price_part["收盤價"] = df_price_part.apply(lambda x : float(x["收盤價"]), axis = 1)

df_price_part = df_price_part.pivot(index = "日期", columns = "股票代號", values = "收盤價")
df_price_part = df_price_part.fillna(method = "ffill")
df_price_part = df_price_part.reset_index(drop = False)

df_price = pd.concat([df_price, df_price_part], axis = 0)
df_price = df_price.fillna(method = "ffill")
df_price = df_price.reset_index(drop = True)

#ret
df_ret_part = df[df["收盤價"] != ""].reset_index(drop = True)
df_ret_part["收盤價"] = df_ret_part.apply(lambda x : float(x["收盤價"]), axis = 1)
df_ret_part["漲跌"] = df_ret_part.apply(lambda x : float(x["漲跌"]), axis = 1)

df_ret_part["ret"] = df_ret_part["漲跌"] / (df_ret_part["收盤價"] - df_ret_part["漲跌"])

df_ret_part = df_ret_part.pivot(index = "日期", columns = "股票代號", values = "ret")
df_ret_part = df_ret_part.fillna(0)
df_ret_part = df_ret_part.reset_index(drop = False)

df_ret = pd.concat([df_ret, df_ret_part], axis = 0)    
df_ret = df_ret.reset_index(drop = True)

#delta
df_delta_part = df[df["總市值(億)"] != ""].reset_index(drop = True)
df_delta_part["總市值"] = df_delta_part.apply(lambda x : float(x["總市值(億)"]), axis = 1)

df_delta_part = df_delta_part.pivot(index = "日期", columns = "股票代號", values = "總市值")
df_delta_part = df_delta_part.fillna(method = "ffill")
df_delta_part = df_delta_part.reset_index(drop = False)

df_delta = pd.concat([df_delta, df_delta_part], axis = 0)
df_delta = df_delta.reset_index(drop = True)

#vol
df_vol_part = df[df["成交金額(千)"] != ""].reset_index(drop = True)
df_vol_part["成交金額"] = df_vol_part.apply(lambda x : float(x["成交金額(千)"]), axis = 1)

df_vol_part = df_vol_part.pivot(index = "日期", columns = "股票代號", values = "成交金額")
df_vol_part = df_vol_part.fillna(method = "ffill")
df_vol_part = df_vol_part.reset_index(drop = False)

df_vol = pd.concat([df_vol, df_vol_part], axis = 0)
df_vol = df_vol.reset_index(drop = True)

del dd, df_delta_part, df_price_part, df_ret_part, df_vol_part, mm, yy


#建模用的資料
df_model_price = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\or_price.csv")
df_model_price["日期"] = pd.to_datetime(df_model_price["日期"], format = "%Y-%m-%d")

df = "select 日期, 股票代號, 股票名稱, 收盤價 from  [dbo].[日收盤還原表排行] where (日期 > '" + start_day + "') and (len(股票代號) = 4)"
sqltables = "日收盤還原表排行"
df = PX.Sql_data(df, sqltables)

df = df[df.apply(lambda x : len(x["股票代號"]), axis = 1) == 4] #留下4碼的股票代號
df = df[df["股票代號"] >= "1101"].reset_index(drop = True)

df["日期"] = pd.to_datetime(df["日期"], format = "%Y%m%d")

#price
df_price_part = df[df["收盤價"] != ""].reset_index(drop = True)
df_price_part["收盤價"] = df_price_part.apply(lambda x : float(x["收盤價"]), axis = 1)

df_price_part = df_price_part.pivot(index = "日期", columns = "股票代號", values = "收盤價")
df_price_part = df_price_part.fillna(method = "ffill")
df_price_part = df_price_part.reset_index(drop = False)

df_model_price = pd.concat([df_model_price, df_price_part], axis = 0)
df_model_price = df_model_price.fillna(method = "ffill")
df_model_price = df_model_price.reset_index(drop = True)

del df, df_price_part


# df_price.to_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\price1218.csv", index = False)
# df_delta.to_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\delta1218.csv", index = False)
# df_ret.to_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\ret1218.csv", index = False)
# df_vol.to_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\vol1218.csv", index = False)

# df_model_price.to_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\or_price1218.csv", index = False)



#%%


df_output = pd.DataFrame()
df_stock_300 = pd.DataFrame()

start_day2 = df_price[df_price["日期"] > pd.to_datetime(start_day, format = "%Y%m%d")].index[0]


for t in range(start_day2, len(df_price)):
    
    #先找出當日前三百大市值
    stock_300 = df_delta[df_delta["日期"] == df_price["日期"].iloc[t]].T.iloc[1:].reset_index(drop = False).sort_values(by = t, ascending = False).iloc[:300].reset_index(drop = True)
    stock_300.columns = ["股票代號", "市值"]
    stock_300["日期"] = df_price["日期"].iloc[t]
    
    df_stock_300 = pd.concat([df_stock_300, stock_300], axis = 0)

    for x in stock_300["股票代號"]:
        
        st_name = x
        
        df_stock = df_model_price[["日期", st_name]]
        
        df_stock["日期"] = pd.to_datetime(df_stock["日期"], format = "%Y%m%d")
        df_stock = df_stock.sort_values(by = "日期")
        
        df_stock["收盤價"] = df_stock.apply(lambda x : float(x[st_name]), axis = 1)
        df_stock = df_stock.dropna(axis = 0)
        
        df_stock = df_stock[df_stock["日期"] <= df_price["日期"].iloc[t]]
        
        df_stock = df_stock.reset_index(drop = True)
        df_stock = df_stock.reset_index(drop = False) #產生x
        
        if(len(df_stock) >= 252*3):
    
            df_part = df_stock.iloc[-(252*3):].reset_index(drop = True)
            
            x_train = np.array(df_part["index"]).reshape(-1, 1)
            y_train = np.array(df_part['收盤價']).reshape(-1, 1)
        
            lm = LinearRegression()
            df_m = lm.fit(x_train, y_train)
            
            yhat = df_m.coef_[0][0] * df_part["index"] + df_m.intercept_[0] #因為要留下beta alpha 所以不用predict 改用這種方式
            yhat_up = df_m.coef_[0][0] * df_part["index"] + df_m.intercept_[0] #因為要留下beta alpha 所以不用predict 改用這種方式
        
            #起始beta
            beta_final = df_m.coef_[0][0]
            alpha_final = df_m.coef_[0][0]
            
            befor_line_up = len(df_part[df_part["收盤價"] >= yhat_up]) #前一次次數
        
            while((len(df_part[df_part["收盤價"] >= yhat_up]) <= befor_line_up) and (len(df_part[df_part["收盤價"] >= yhat_up]) >= 10)):
                
                df_part2 = df_part[df_part["收盤價"] >= yhat_up].reset_index(drop = True)
                
                x_train = np.array(df_part2["index"]).reshape(-1, 1)
                y_train = np.array(df_part2['收盤價']).reshape(-1, 1)
        
                df_m_up = lm.fit(x_train, y_train)
                
                yhat_up = df_m_up.coef_[0][0] * df_part["index"] + df_m_up.intercept_[0] #因為要留下beta alpha 所以不用predict 改用這種方式
                
                #大於上界的人有變少才紀錄
                if(len(df_part[df_part["收盤價"] >= yhat_up]) <= befor_line_up):
                    
                    befor_line_up = len(df_part[df_part["收盤價"] >= yhat_up])
                    
                    #更新beta
                    beta_final = df_m_up.coef_[0][0]
                    alpha_final = df_m_up.intercept_[0]
                
            yhat_up = beta_final * df_part["index"] + alpha_final #因為要留下beta alpha 所以不用predict 改用這種方式
                
            df_output_part = pd.DataFrame([df_part["日期"].iloc[-1], df_part["收盤價"].iloc[-1], yhat_up.iloc[-1], st_name]).T
            df_output = pd.concat([df_output, df_output_part], axis = 0)
        

df_output.columns = ["日期", "收盤價", "出場價", "股票代號"]
df_output_up_price_part = df_output.pivot(columns = "股票代號", index = "日期", values = "出場價")
df_output_up_price_part = df_output_up_price_part.reset_index(drop = False)

df_output_up_price = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0319_趨勢判斷\\data\\df_output_up_price.csv")
df_output_up_price["日期"] = pd.to_datetime(df_output_up_price["日期"], format = "%Y-%m-%d")

df_output_up_price = pd.concat([df_output_up_price, df_output_up_price_part], axis = 0)
df_output_up_price = df_output_up_price.reset_index(drop = True)


#%%
#找訊號

df_sign = pd.DataFrame()

for x in df_output_up_price.columns[1:]:
    
    df_sign_part = pd.merge(df_output_up_price[["日期", x]], df_model_price[["日期", x]], how = "left", on = "日期")
    df_sign_part.columns = ["日期", "up", "price"]

    df_sign_part["sign1"] = df_sign_part.apply(lambda x : 1 if x["price"] > x["up"] else 0, axis = 1)


    #第一次進場
    sign2 = [df_sign_part["sign1"].iloc[0]]
    
    for i in range(1,len(df_sign_part)):
        
        #第一次進場
        if((df_sign_part["sign1"].iloc[i] == 1) and (df_sign_part["sign1"].iloc[i-1] == 0)):
            
            sign2.append(1)
            
        #0和續抱都先改0
        else:
            
            sign2.append(0)
        
        
    df_sign_part["sign2"] = sign2
    
    del i
    
    
    #判定續抱 or 出場 or 重新進場
    sign3 = [df_sign_part["sign2"].iloc[0]]
    price_test = [0]
    
    for i in range(1, len(df_sign_part)):
            
        #情況一 純進場 產生停損價格
        if((df_sign_part["sign2"].iloc[i] == 1) and (sign3[i-1] == 0)):
            
            stop_price = df_sign_part["price"].iloc[i] * 0.95
            sign3.append(1)
            
        #情況二 純續抱 判斷出場 或 產生新的停損價格
        if((sign3[i-1] == 1)):
            
            #先判斷是否出場 跌破停損價格 就出場
            if(df_sign_part["price"].iloc[i] < stop_price):
                
                stop_price = price_test[i-1]
                sign3.append(0)
            
            #續抱 價格更新
            else:
                
                #如果是下跌 價格不更新
                if(df_sign_part["price"].iloc[i] <= df_sign_part["price"].iloc[i-1]):
                    
                    stop_price = price_test[i-1]
                
                #如果上漲 則用上漲的新價格計算停損價格
                else:
                    
                    #線算上漲的話的新價格
                    stop_price = df_sign_part["price"].iloc[i] * 0.95
                    
                    #如果新價格低於前一次用的用前一次
                    #有實際超過才更新
                    if(stop_price <= price_test[i-1]):
                        
                        stop_price = price_test[i-1]
                    
                sign3.append(1)
                
        #情況三 不進場 不續抱
        if((df_sign_part["sign2"].iloc[i] == 0) and (sign3[i-1] == 0)):
            
            stop_price = price_test[i-1]
            sign3.append(0)
            
        price_test.append(stop_price)
            
    
    df_sign_part["sign3"] = sign3
    df_sign_part["stop_price"] = price_test   

    df_sign_part["股票代號"] = x

    df_sign = pd.concat([df_sign, df_sign_part], axis = 0)


df_sign_output = df_sign.pivot(columns = "股票代號", index = "日期", values = "sign3")
df_sign_output = df_sign_output.reset_index(drop = False)

#持有名單
hold_list = df_sign_output.iloc[-2:].T
hold_list = hold_list.iloc[1:].reset_index(drop = False)
hold_list.columns = ["股票代號", "昨天", "今天"]
hold_list = hold_list[hold_list["今天"] == 1].reset_index(drop = True)



#找出進場日期 與刪掉下市

infor_for_holding_stock = pd.DataFrame()

for i in range(0, len(hold_list)):
    
    st_name = hold_list["股票代號"].iloc[i]
    hold_st = df_sign_output[["日期", st_name]]
    
    final_date_in_not_hold = hold_st[hold_st[st_name] == 0]["日期"].iloc[-1]
    start_date_in_hold = hold_st[hold_st["日期"] > final_date_in_not_hold]["日期"].iloc[0]
    
    start_date_price = df_price[df_price["日期"] == start_date_in_hold][st_name].iloc[0]
    
    infor_for_holding_stock_part = pd.DataFrame([st_name, start_date_in_hold, start_date_price]).T
    
    infor_for_holding_stock = pd.concat([infor_for_holding_stock, infor_for_holding_stock_part], axis = 0)

infor_for_holding_stock.columns = ["股票代號", "進場日期", "進場價格"]

df = "select 日期, 股票代號, 股票名稱, 收盤價 from  [dbo].[日收盤還原表排行] where (日期 > '" + start_day + "') and (len(股票代號) = 4)"
sqltables = "日收盤還原表排行"
df = PX.Sql_data(df, sqltables)

st_name_list = df[["股票代號", "股票名稱"]]
st_name_list = st_name_list.drop_duplicates(subset = ["股票代號"])


df = df.pivot(columns = "股票代號", index = "日期", values = "收盤價")
df = df.fillna(method = "ffill")
df = df.iloc[-1]

df = df.reset_index(drop = False)
df["上市中"] = 1

infor_for_holding_stock = pd.merge(infor_for_holding_stock, df, how = "left", on = "股票代號")

infor_for_holding_stock = infor_for_holding_stock.dropna(axis = 0)
infor_for_holding_stock = pd.merge(infor_for_holding_stock, st_name_list, how = "left", on = "股票代號")

infor_for_holding_stock = pd.merge(infor_for_holding_stock, hold_list, how = "left", on = "股票代號")

infor_for_holding_stock["庫存情形"] = infor_for_holding_stock.apply(lambda x : "新進" if (x["今天"] - x["昨天"]) == 1 else "續抱", axis = 1)

infor_for_holding_stock.columns = ["股票代號", "進場日期", "進場價格", "今收", "上市情形", "股票名稱", "昨訊號", "今訊號", "庫存情形"]
infor_for_holding_stock = infor_for_holding_stock[["股票代號", "股票名稱", "進場日期", "進場價格", "今收", "昨訊號", "今訊號", "庫存情形"]]





df_ret_output = df_ret[df_sign_output.columns][df_ret[df_sign_output.columns]["日期"] >= "2017/2/8"]



