import os
from xml import dom
import pandas as pd
import shutil

def dir2df(df, source_dir,domain,train_split=None):
    src_class_dir = os.path.join(source_dir,domain)
    tmp_df=pd.DataFrame({"Files":os.listdir(src_class_dir)})
    df =df.append(tmp_df,ignore_index=True)
    

    return df

def annotate_df(df,source_dir):
    df[['Subject_ID',"Pose_ID","Class","Modality"]] = df["Files"].str.split('_', expand=True)
    df["Modality"]=df["Modality"].apply(lambda x: x.split(".")[0])
    df["Subject_ID"]=df["Subject_ID"].astype(int)
    df["Pose_ID"] = df["Pose_ID"].astype(int)
    df["Path"] = df.apply(lambda x: os.path.join(source_dir,x["Class"].capitalize(),x["Files"]),axis=1)
    
    return df

def split_df(df,train_split=None):
    if train_split is not None:
        subjects= df["Subject_ID"].unique()
        subjects.sort()
        train_subjects = subjects[:int(len(subjects)*train_split)]
        test_subjects = subjects[int(len(subjects)*train_split):]

        df.loc[df['Subject_ID'].isin(train_subjects),"Split"]="train"
        df.loc[df['Subject_ID'].isin(test_subjects),"Split"]="val"
    return df
def create_dirs(df,dest_dir):
    for split in df["Split"].unique():
            for domain in df["Class"].unique():
                    dest_split_dir= os.path.join(dest_dir,split,domain)
                    os.makedirs(dest_split_dir,exist_ok=True)
    


if __name__=="__main__":
    source_dir = "/home/shehzikhan/Projects/Fresh WorkSpace/sejong_db/Data"
    dest_dir = "/home/shehzikhan/Projects/Fresh WorkSpace/stargan-v2/data/sejong_db"
    domains = ["Glasses","Normal"]
    train_split=0.8
    df = pd.DataFrame({"Files":[]})

    for domain in os.listdir(source_dir):
        if domain in domains:
            df = dir2df(df,source_dir,domain,train_split)
            # df = df.sort_values(by=['Subject_ID'])
    
    df = annotate_df(df,source_dir)
    df = split_df(df,train_split)
    if "Split" in df.columns:
        create_dirs(df,dest_dir)
        for idx,row in df.iterrows():
            if not os.path.isfile(os.path.join(dest_dir,row["Split"],row["Class"],row["Files"])):
                shutil.copy(row["Path"],os.path.join(dest_dir,row["Split"],row["Class"],row["Files"]))
    print(df)





