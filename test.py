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
#資料更新
df_price = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\price2.csv")
df_delta = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\delta2.csv")
df_ret = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\ret2.csv")
df_vol = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\vol2.csv")

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

#更新
df = "select 日期, 股票代號, 股票名稱, 收盤價, 漲跌, [成交金額(千)], [總市值(億)] from  [dbo].[日收盤表排行] where (日期 > '" + start_day + "') and (len(股票代號) = 4)"
sqltables = "日收盤表排行"
df = PX.Sql_data(df, sqltables)

df = df[df.apply(lambda x : len(x["股票代號"]), axis = 1) == 4] #留下4碼的股票代號
df = df[df["股票代號"] >= "1101"].reset_index(drop = True)

df["日期"] = pd.to_datetime(df["日期"], format = "%Y%m%d")

#公司名稱
df_company = "select 日期, 股票代號, 股票名稱 from  [dbo].[日收盤表排行] where (日期 >= '" + start_day + "') and (len(股票代號) = 4)"
sqltables = "日收盤表排行"
df_company = PX.Sql_data(df_company, sqltables)
df_company = df_company.drop_duplicates(subset = ["股票代號"], keep = "last")

#price
df_price_part = df[df["收盤價"] != ""].reset_index(drop = True)
df_price_part["收盤價"] = df_price_part.apply(lambda x : float(x["收盤價"]), axis = 1)

df_price_part = df_price_part.pivot(index = "日期", columns = "股票代號", values = "收盤價")
df_price_part = df_price_part.reset_index(drop = False)

df_price = pd.concat([df_price, df_price_part], axis = 0)
df_price = df_price.reset_index(drop = True)

#ret
df_ret_part = df[df["收盤價"] != ""].reset_index(drop = True)
df_ret_part["收盤價"] = df_ret_part.apply(lambda x : float(x["收盤價"]), axis = 1)
df_ret_part["漲跌"] = df_ret_part.apply(lambda x : float(x["漲跌"]), axis = 1)

df_ret_part["ret"] = df_ret_part["漲跌"] / (df_ret_part["收盤價"] - df_ret_part["漲跌"])

df_ret_part = df_ret_part.pivot(index = "日期", columns = "股票代號", values = "ret")
df_ret_part = df_ret_part.reset_index(drop = False)

df_ret = pd.concat([df_ret, df_ret_part], axis = 0)    
df_ret = df_ret.reset_index(drop = True)

#delta
df_delta_part = df[df["總市值(億)"] != ""].reset_index(drop = True)
df_delta_part["總市值"] = df_delta_part.apply(lambda x : float(x["總市值(億)"]), axis = 1)

df_delta_part = df_delta_part.pivot(index = "日期", columns = "股票代號", values = "總市值")
df_delta_part = df_delta_part.reset_index(drop = False)

df_delta = pd.concat([df_delta, df_delta_part], axis = 0)
df_delta = df_delta.reset_index(drop = True)

#vol
df_vol_part = df[df["成交金額(千)"] != ""].reset_index(drop = True)
df_vol_part["成交金額"] = df_vol_part.apply(lambda x : float(x["成交金額(千)"]), axis = 1)

df_vol_part = df_vol_part.pivot(index = "日期", columns = "股票代號", values = "成交金額")
df_vol_part = df_vol_part.reset_index(drop = False)

df_vol = pd.concat([df_vol, df_vol_part], axis = 0)
df_vol = df_vol.reset_index(drop = True)

del dd, df_delta_part, df_price_part, df_ret_part, df_vol_part, mm, yy


#建模用的資料
df_model_price = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\or_price2.csv")
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

if(len(df_price_part) > 1):
    
    df_price_part = df_price_part.reset_index(drop = False)
    df_price_part = df_price_part[df_price_part["日期"] > df_model_price["日期"].iloc[-1]].reset_index(drop = False)

if(len(df_price_part) == 1):

    df_price_part = df_price_part.reset_index(drop = False)


df_model_price = pd.concat([df_model_price, df_price_part], axis = 0)
df_model_price = df_model_price.reset_index(drop = True)

del df, df_price_part


#下市清單
df_close_company = "select 年度, 股票代號, 股票名稱, 終止日期 from  [dbo].[下市櫃公司基本資料] where (len(股票代號) = 4)"
sqltables = "下市櫃公司基本資料"
df_close_company = PX.Sql_data(df_close_company, sqltables)
df_close_company["日期"] = pd.to_datetime(df_close_company["終止日期"], format = "%Y%m%d")

#%%
#找出上界
df_output_up_price = pd.read_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\df_output_up_price2.csv")
df_output_up_price["日期"] = pd.to_datetime(df_output_up_price["日期"], format = "%Y-%m-%d")

df_output = pd.DataFrame()
df_stock_300 = pd.DataFrame()

t_start = df_price[df_price["日期"] == df_output_up_price["日期"].iloc[-1]].index[0] + 1

for t in range(t_start, len(df_price)):
    
    #先找出當日前三百大市值
    stock_300 = df_delta[df_delta["日期"] == df_price["日期"].iloc[t]].T.iloc[1:].reset_index(drop = False).sort_values(by = t, ascending = False).iloc[:500].reset_index(drop = True)
    stock_300.columns = ["股票代號", "市值"]
    stock_300["日期"] = df_price["日期"].iloc[t]
    
    df_stock_300 = pd.concat([df_stock_300, stock_300], axis = 0)

    for x in stock_300["股票代號"]:
        
        st_name = x
        
        df_stock = df_model_price[["日期", st_name]]
        
        df_stock["日期"] = pd.to_datetime(df_stock["日期"], format = "%Y%m%d")
        df_stock = df_stock.sort_values(by = "日期")
        df_stock[st_name] = df_stock[st_name].fillna(method = "ffill")
        
        df_stock["收盤價"] = df_stock.apply(lambda x : float(x[st_name]), axis = 1)
        df_stock = df_stock.dropna(axis = 0)
        
        df_stock = df_stock[df_stock["日期"] <= df_price["日期"].iloc[t]]
        
        df_stock = df_stock.reset_index(drop = True)
        df_stock = df_stock.reset_index(drop = False) #產生x
        
        #判定是否下市
        close_yes = df_close_company[df_close_company["股票代號"] == st_name]
        
        if(len(close_yes) == 1):
            
            df_stock = df_stock[df_stock["日期"] <= close_yes["日期"].iloc[0]]
        
        
        if(df_stock["日期"].iloc[-1] >= df_price["日期"].iloc[t]):
            
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
df_output_up_price_part["日期"] = pd.to_datetime(df_output_up_price_part["日期"], format = "%Y-%m-%d")

df_output_up_price = pd.concat([df_output_up_price, df_output_up_price_part], axis = 0)


del alpha_final, befor_line_up, beta_final, close_yes, df_m, df_m_up, df_output, df_output_part, df_output_up_price_part
del df_part, df_part2, lm, start_day, stock_300, t, t_start, x, x_train, y_train, yhat, yhat_up


#%%
start = 252*3
end = len(df_model_price)

df_sign = pd.DataFrame()
df_cost = pd.DataFrame()
df_stop_price = pd.DataFrame()

for t in range(start, end):
    
    df_part = df_model_price.iloc[t].reset_index(drop = False).iloc[1:]
    df_part.columns = ["股票代號", t]
    df_part = pd.merge(df_part, df_output_up_price.iloc[t - 252*3].reset_index(drop = False).rename(columns = {"index" : "股票代號"}), how = "left", on = "股票代號")
    
    df_part.columns = ["股票代號", "price", "up"]
    
    df_part["sign"] = df_part.apply(lambda x : 1 if x["price"] > x["up"] else 0, axis = 1)
    df_part = df_part.dropna()

    if(len(df_sign) == 0):
        
        df_part["cost"] = df_part["price"] * df_part["sign"]
        df_part["stop"] = df_part["price"] * df_part["sign"] * 0.95
        
        df_part = df_part.T
        df_part.columns = df_part.iloc[0]
        
        df_sign_part = pd.DataFrame(df_part.iloc[3]).T
        df_cost_part = pd.DataFrame(df_part.iloc[4]).T
        df_stop_price_part = pd.DataFrame(df_part.iloc[5]).T
        
        df_sign_part["日期"] = df_model_price["日期"].iloc[t]
        df_cost_part["日期"] = df_model_price["日期"].iloc[t]
        df_stop_price_part["日期"] = df_model_price["日期"].iloc[t]
    
        df_sign = pd.concat([df_sign, df_sign_part], axis = 0) 
        df_cost = pd.concat([df_cost, df_cost_part], axis = 0) 
        df_stop_price = pd.concat([df_stop_price, df_stop_price_part], axis = 0) 
        
        #處理columns排序
        stock_columns = ["日期"]
        stock_columns.extend(df_sign.columns[:-1].to_list())
        
        df_sign = df_sign[stock_columns]
        df_cost = df_cost[stock_columns]
        df_stop_price = df_stop_price[stock_columns]
        
    
    else:
        
        #加入未在市值300 但有持股的人要加回
        outer_300_have_stock = df_sign.iloc[-1].iloc[1:].reset_index(drop = False)
        outer_300_have_stock.columns = ["股票代號", "b_sign"]
        outer_300_have_stock = outer_300_have_stock[outer_300_have_stock["b_sign"] == 1]
        outer_300_have_stock = outer_300_have_stock[outer_300_have_stock["股票代號"].isin(df_part["股票代號"]) == False]
        
        outer_300_have_stock_or_price = df_model_price.iloc[t].reset_index(drop = False)
        outer_300_have_stock_or_price.columns = ["股票代號", t]
        
        outer_300_have_stock = pd.merge(outer_300_have_stock, outer_300_have_stock_or_price, how = "left", on = "股票代號")
        outer_300_have_stock.columns = ["股票代號", "前日訊號", "price"]
        
        outer_300_have_stock = outer_300_have_stock[["股票代號", "price"]]
            
        df_part = pd.concat([df_part, outer_300_have_stock], axis = 0).reset_index(drop = True)
        
        #再次判定是否下市
        close_drop_list = []
        for k in range(0,len(df_part)):
            
            st_name = df_part["股票代號"].iloc[k]
            close_yes = df_close_company[df_close_company["股票代號"] == st_name]
            
            if(len(close_yes) == 1):
                
                if(df_model_price["日期"].iloc[t] > close_yes["日期"].iloc[0]):
                    close_drop_list.append(st_name)
        
        df_part = df_part[df_part["股票代號"].isin(close_drop_list) == False]
        
        #找出前一天資訊(收盤 訊號 停損 成本)
        b_price = df_model_price.iloc[t-1].reset_index(drop = False).iloc[1:]
        b_price.columns = ["股票代號", "b_price"]
        
        b_sign = df_sign.iloc[t - (start + 1)].reset_index(drop = False).iloc[1:]
        b_sign.columns = ["股票代號", "b_sign"]
        
        b_stop_price = df_stop_price.iloc[t - (start + 1)].reset_index(drop = False).iloc[1:]
        b_stop_price.columns = ["股票代號", "b_stop"]
        
        b_cost = df_cost.iloc[t - (start + 1)].reset_index(drop = False).iloc[1:]
        b_cost.columns = ["股票代號", "b_cost"]
        
        df_part = pd.merge(df_part, b_price, how = "left", on = "股票代號") #收盤
        df_part = pd.merge(df_part, b_sign, how = "left", on = "股票代號") #訊號
        df_part = pd.merge(df_part, b_stop_price, how = "left", on = "股票代號") #訊號
        df_part = pd.merge(df_part, b_cost, how = "left", on = "股票代號") #訊號
        
        #判定今天的訊號
        
        new_sign = []
        sign_stock = []
        
        for i in range(0,len(df_part)):
        
            #step 1 第一次進場
            if(df_part["sign"].iloc[i] == 1 and (df_part["b_sign"].iloc[i] == 0 or np.isnan(df_part["b_sign"].iloc[i]))):
                new_sign.append(1)
                sign_stock.append(df_part["股票代號"].iloc[i])
            
            #step2 前一天0或na 跟 今天都是0
            if(df_part["sign"].iloc[i] == 0 and (df_part["b_sign"].iloc[i] == 0 or np.isnan(df_part["b_sign"].iloc[i]))):
                new_sign.append(0)   
                sign_stock.append(df_part["股票代號"].iloc[i])
            
            #step3 前一天進場 今天為0 判定是否續抱
            if(df_part["sign"].iloc[i] == 0 and df_part["b_sign"].iloc[i] == 1):
                
                #純跌破出場
                if(df_part["price"].iloc[i] <= df_part["b_stop"].iloc[i]):
                    new_sign.append(0)
                    sign_stock.append(df_part["股票代號"].iloc[i])
            
                #沒跌破 續抱
                if(df_part["price"].iloc[i] > df_part["b_stop"].iloc[i]):
                    new_sign.append(1)
                    sign_stock.append(df_part["股票代號"].iloc[i])
                    
            #step4 都是進場 續抱
            if(df_part["sign"].iloc[i] == 1 and df_part["b_sign"].iloc[i] == 1): 
                new_sign.append(1)
                sign_stock.append(df_part["股票代號"].iloc[i])
                
            #step5 前一天有 但今天未在300大市值 判定是否續抱 (包含有可能休市)
            if(np.isnan(df_part["sign"].iloc[i]) and df_part["b_sign"].iloc[i] == 1):
                
                #純跌破出場
                if(df_part["price"].iloc[i] <= df_part["b_stop"].iloc[i]):
                    new_sign.append(0)
                    sign_stock.append(df_part["股票代號"].iloc[i])
            
                #沒跌破 續抱
                if(df_part["price"].iloc[i] > df_part["b_stop"].iloc[i]):
                    new_sign.append(1)
                    sign_stock.append(df_part["股票代號"].iloc[i])
                    
                #休市
                if(np.isnan(df_part["price"].iloc[i]) and (np.isnan(df_part["b_sign"].iloc[i]) == False)):
                    new_sign.append(df_part["b_sign"].iloc[i])
                    sign_stock.append(df_part["股票代號"].iloc[i])

            
        new_sign = pd.DataFrame(new_sign)
        sign_stock = pd.DataFrame(sign_stock)
        
        new_sign = pd.concat([sign_stock, new_sign], axis = 1)
        new_sign.columns = ["股票代號", "n_sign"]
            
        df_part = pd.merge(df_part, new_sign, how = "left", on = "股票代號")
        
        #更新成本與新的出場價格
        new_cost = []
        new_stop_price = []
        sign_stock = []
        
        for i in range(0,len(df_part)):
            
            #沒部位
            if(df_part["n_sign"].iloc[i] == 0):
                
                #前日也沒有 均為0
                new_cost.append(0)
                new_stop_price.append(0)
                sign_stock.append(df_part["股票代號"].iloc[i])
            
            #有部位
            if(df_part["n_sign"].iloc[i] != 0):
                
                #第一天進場的人 要建立成本 與 停損價
                if(df_part["n_sign"].iloc[i] == 1 and (df_part["b_sign"].iloc[i] == 0 or np.isnan(df_part["b_sign"].iloc[i]))):
                    
                    new_cost.append(df_part["price"].iloc[i])   
                    new_stop_price.append(df_part["price"].iloc[i] * 0.95)
                    sign_stock.append(df_part["股票代號"].iloc[i])
                
                #續抱的人 判定是否更新停損價格
                if(df_part["n_sign"].iloc[i] == 1 and df_part["b_sign"].iloc[i] == 1):
                    
                    new_cost.append(df_part["b_cost"].iloc[i])
                    
                    #下跌不更新出場價格 上升更新出場價格
                    #上升 更新
                    if(df_part["price"].iloc[i] > df_part["b_price"].iloc[i]):
                        
                        #情況一 累計漲幅不超過50% 停損價格一樣為0.95
                        if((df_part["price"].iloc[i] / df_part["b_cost"].iloc[i] - 1) < 0.5):
                            
                            #但如果漲了但停損價格 低於上次的停損價格 則用上次的 (表示先跌再漲 但沒有漲超過)
                            if(df_part["price"].iloc[i] * 0.95 < df_part["b_stop"].iloc[i]):
                            
                                new_stop_price.append(df_part["b_stop"].iloc[i])
                                sign_stock.append(df_part["股票代號"].iloc[i])
                            
                            if(df_part["price"].iloc[i] * 0.95 >= df_part["b_stop"].iloc[i]):
                            
                                new_stop_price.append(df_part["price"].iloc[i] * 0.95)
                                sign_stock.append(df_part["股票代號"].iloc[i])
                                
                        #情況二 累計漲幅超過50% 停損價為0.9
                        if((df_part["price"].iloc[i] / df_part["b_cost"].iloc[i] - 1) >= 0.5):
                            
                            #但如果漲了但停損價格 低於上次的停損價格 則用上次的 (表示先跌再漲 但沒有漲超過)
                            if(df_part["price"].iloc[i] * 0.9 < df_part["b_stop"].iloc[i]):
                            
                                new_stop_price.append(df_part["b_stop"].iloc[i])
                                sign_stock.append(df_part["股票代號"].iloc[i])
                            
                            if(df_part["price"].iloc[i] * 0.9 >= df_part["b_stop"].iloc[i]):
                            
                                new_stop_price.append(df_part["price"].iloc[i] * 0.9)
                                sign_stock.append(df_part["股票代號"].iloc[i])
                    
                    #下跌沒跌破 不更新
                    if(df_part["price"].iloc[i] <= df_part["b_price"].iloc[i]):
                        
                        new_stop_price.append(df_part["b_stop"].iloc[i])
                        sign_stock.append(df_part["股票代號"].iloc[i])
                        
                    #沒開盤 續抱
                    if(np.isnan(df_part["price"].iloc[i]) and (df_part["n_sign"].iloc[i] == 1)):
                        
                        new_stop_price.append(df_part["b_stop"].iloc[i])
                        sign_stock.append(df_part["股票代號"].iloc[i])
                        
                    #休市後開盤 續抱
                    if(np.isnan(df_part["b_price"].iloc[i]) and (df_part["price"].iloc[i] > 0) and (df_part["n_sign"].iloc[i] == 1)):
                        
                        new_stop_price.append(df_part["b_stop"].iloc[i])
                        sign_stock.append(df_part["股票代號"].iloc[i])
                
            
        
        new_cost = pd.DataFrame(new_cost)
        new_stop_price = pd.DataFrame(new_stop_price)
        sign_stock = pd.DataFrame(sign_stock)
        
        sign_stock = pd.concat([sign_stock, new_stop_price, new_cost], axis = 1)
        sign_stock.columns = ["股票代號", "n_stop", "n_cost"]
           
        df_part = pd.merge(df_part, sign_stock, how = "left", on = "股票代號")
        
        #輸出
        df_sign_part = df_part[["股票代號", "n_sign"]]
        df_sign_part = df_sign_part.T
        df_sign_part.columns = df_sign_part.iloc[0]
        df_sign_part = pd.DataFrame(df_sign_part.iloc[1]).T
        df_sign_part["日期"] = df_model_price["日期"].iloc[t]
        
        df_cost_part = df_part[["股票代號", "n_cost"]]
        df_cost_part = df_cost_part.T
        df_cost_part.columns = df_cost_part.iloc[0]
        df_cost_part = pd.DataFrame(df_cost_part.iloc[1]).T
        df_cost_part["日期"] = df_model_price["日期"].iloc[t]
        
        df_stop_price_part = df_part[["股票代號", "n_stop"]]
        df_stop_price_part = df_stop_price_part.T
        df_stop_price_part.columns = df_stop_price_part.iloc[0]
        df_stop_price_part = pd.DataFrame(df_stop_price_part.iloc[1]).T
        df_stop_price_part["日期"] = df_model_price["日期"].iloc[t]
        
        df_sign = pd.concat([df_sign, df_sign_part], axis = 0) 
        df_cost = pd.concat([df_cost, df_cost_part], axis = 0) 
        df_stop_price = pd.concat([df_stop_price, df_stop_price_part], axis = 0) 
        
        
del b_cost, b_price, b_sign, b_stop_price, close_yes, close_drop_list, df_cost_part, df_part, df_sign_part, df_stop_price_part, end, i, k, new_cost, new_sign, new_stop_price
del outer_300_have_stock, outer_300_have_stock_or_price, sign_stock, start, stock_columns, t


#%%
#找出今天進場的人
df_new = df_sign.iloc[-2:].T.iloc[1:]
df_new.columns = ["昨", "今"]
df_new["new"] = df_new["今"] - df_new["昨"]

df_new = pd.merge(df_new, df_company[["股票代號", "股票名稱"]], how = "left", on = "股票代號")



#%%
# df_price.to_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\price2.csv", index = False)
# df_delta.to_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\delta2.csv", index = False)
# df_ret.to_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\ret2.csv", index = False)
# df_vol.to_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\vol2.csv", index = False)
# df_model_price.to_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\or_price2.csv", index = False)
# df_output_up_price.to_csv("D:\\Python_code\\工作內容\\2024\\0411_強弱勢Tpad\\data\\df_output_up_price2.csv", index = False)


# # #%%
# df_ret2 = df_ret[df_sign.columns]
# df_ret2 = df_ret2[df_ret2["日期"] >= df_sign["日期"].iloc[0]]


















