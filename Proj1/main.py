import dataprocessing as dp
import modelcreator as mc
import TestMe

df = dp.clean_data_set("Lab6Dataset.csv")
model = mc.create_model(df)
TestMe.run_test(model)

