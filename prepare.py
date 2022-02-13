def prep_iris(df):
    df.drop(columns='species_id', inplace=True)
    df.rename(columns={'species_name':'species'}, inplace=True)
    dummy_df = pd.get_dummies(df['species'], drop_first=True)
    iris = pd.concat([df, dummy_df], axis=1)
    return iris

def prep_titanic(df):
    columns_to_drop = ['age', 'deck', 'parch', 'embarked']
    df.drop(columns_to_drop, inplace=True, axis=1)
    categorical_columns = df.select_dtypes(include='object').columns
    dummy_df = pd.get_dummies(df[categorical_columns], drop_first=True)
    titanic = pd.concat([df, dummy_df], axis=1)
    return titanic

def prep_telco(df):
    columns_to_drop = ['payment_type_id', 'internet_service_type_id', 'contract_type_id']
    df.drop(columns_to_drop, axis=1, inplace=True)
    categorical_columns = df.select_dtypes(include='object').drop(columns='customer_id')
    dummy_df = pd.get_dummies(categorical_columns, drop_first=True)
    telco = pd.concat([df, dummy_df], axis=1)
    return telco