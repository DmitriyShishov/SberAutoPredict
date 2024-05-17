import dill
import json

import pandas as pd


def main():
    with open('model/sber_auto_prod.pkl', "rb") as file:
        model = dill.load(file)

    print(model['metadata'])

    with open('jsons/Example_1_Target_0.json', "rb") as fin:
        form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model['model'].predict(df)
        print(f'{form['client_id']}: {y[0]}')

    with open('jsons/Example_2_Target_1.json', "rb") as fin:
        form = json.load(fin)
        df = pd.DataFrame.from_dict([form])
        y = model['model'].predict(df)
        print(f'{form['client_id']}: {y[0]}')




if __name__ == '__main__':
    main()