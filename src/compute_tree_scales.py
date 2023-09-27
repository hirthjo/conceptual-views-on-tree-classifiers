import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import tree
import json 
import sys, os
import argparse
import subprocess
from subprocess import Popen, PIPE
#import multiprocessing as mp
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

CV_NUM = 4 # 4 0 for fit to the whole data set and scaling only
SCORING_MEASURES = ("accuracy","roc_auc")
NUM_CPU = 2

# this file contains a converter of dtrees to scales and evaluates their quality
# Only single class classification is supported for the scaling error
# For multi-class support check the extract_tree method

################# IO
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64,np.longlong)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_data(args):
    '''Loads the data and encodes non-numeric features using
    one-hot-encoding (nominal scaling). If the data set path provided, the
    data set will be read from that file. Otherwise the data set is read
    from the cc18 directory at "../../data/"'''
    if os.path.isfile(args.data):
        filename = args.data
    elif os.path.isfile("../data/%s.csv"%(args.data)):
        filename = "../data/%s.csv"%(args.data)
    else: 
        filename = "../../data/%s.csv"%(args.data)
    data =  pd.read_csv(filename)
    data.dropna(inplace=True)
    data.rename(lambda s: s.replace(" ","_"), axis='columns',inplace=True)
    y = data.pop(args.target)
    X = transform_data(data,y)
    print("Loaded %s Data set of size %i x %i"%(args.data,X.shape[0],X.shape[1]))
    return X,y

def extract_tree(dt, feature_names, classes):
    '''Extracts the predicates annotated to the decision tree leafs. The
    predicate of a leaf is the conjunctive conjunction of the predicates
    on the path. Only single class classification is supported.'''
    feature_name = [feature_names[i] if i != tree._tree.TREE_UNDEFINED else "undefined!" for i in dt.feature]
    def get_paths(path_list, leafs):
        if len(path_list) == 0:
            return leafs
        rest = path_list[1:]
        path = path_list[0]
        dnode = path[-1]
        lchild = dt.children_left[dnode["id"]]
        rchild = dt.children_right[dnode["id"]]
        if dt.feature[lchild] != tree._tree.TREE_UNDEFINED:
            lpath = path[:-1].copy()
            lpath.append({**dnode, "t<v" : True}) # <= threshold
            feature = feature_name[lchild]
            threshold = dt.threshold[lchild]
            lpath.append({"feature": feature, "id" : lchild, "threshold": threshold})
            rest.append(lpath)
        else: 
            lpath = path[:-1].copy()
            lpath.append({**dnode, "t<v" : True})
            lpath.append({"value" : dt.value[lchild][0], "classes" : classes}) # only single class classification considered
            leafs.append(lpath)
        if dt.feature[rchild] != tree._tree.TREE_UNDEFINED:
            rpath = path[:-1].copy()
            rpath.append({**dnode, "t<v" : False}) # > threshold
            feature = feature_name[rchild]
            threshold = dt.threshold[rchild]
            rpath.append({"feature": feature, "id" : rchild, "threshold": threshold})
            rest.append(rpath)
        else: 
            rpath = path[:-1].copy()
            rpath.append({**dnode, "t<v" : False})
            rpath.append({"value" : dt.value[rchild][0], "classes" : classes})
            leafs.append(rpath)
        return get_paths(rest, leafs)
    root = 0
    feature = feature_name[root]
    threshold = dt.threshold[root].item()
    return get_paths([[{"feature": feature, "id" : root, "threshold": threshold}]],[])


def dump_dleafs(dleafs,args,i=1):
    with open(f"../output/{args.data}/{args.mode}/{i}/{args.data}_dleafs.json", 'w', encoding='utf-8') as f:
        json.dump(dleafs, f, cls=NumpyEncoder)

def dump_transformed_split_data(X_train, X_test, y_train, y_test,args):
    print("Dump encoded Tree for clojure")
    out_train_data = pd.concat([X_train,y_train],axis=1).rename(lambda s: s.replace(" ","_"), axis='columns')
    out_train_data.to_csv(f"../output/{args.data}/{args.mode}/{args.data}_train_enc.csv",index=False)
    out_test_data = pd.concat([X_test,y_test],axis=1).rename(lambda s: s.replace(" ","_"), axis='columns')
    out_test_data.to_csv(f"../output/{args.data}/{args.mode}/{args.data}_test_enc.csv",index=False)

def dump_transformed_data(X, y, args):
    print("Dump encoded Tree for clojure")
    out_data = pd.concat([X,y],axis=1).rename(lambda s: s.replace(" ","_"), axis='columns')
    out_data.to_csv(f"../output/{args.data}/{args.mode}/{args.data}_enc.csv",index=False)

################# Pre-processing


def transform_data(X,y):
    '''encode non numeric features (inclusive boolean) using OneHotEncoder'''
    numeric_features = [c for c in X.columns if is_numeric_dtype(X[c]) 
                                                and not is_bool_dtype(X[c])]
    non_numeric_features = [c for c in X.columns if not is_numeric_dtype(X[c]) 
                                                       or is_bool_dtype(X[c])]
    print(f"{len(numeric_features)} numeric features and {len(non_numeric_features)} non numeric")
    numeric_X = X[numeric_features]
    non_numeric_X = X[non_numeric_features]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(non_numeric_X)
    # transform X
    X_enc = enc.transform(non_numeric_X)
    X_enc = pd.DataFrame(X_enc.toarray(),columns=enc.get_feature_names(non_numeric_X.columns))
    transformed_X = pd.concat([numeric_X, X_enc],axis=1)
    return transformed_X



################# Training
def train_tree(X,y, args):
    '''Trains the classifier and dumps the tree structure in a json
    file. This file is then read by conexp-clj.'''
    # train classifier
    if args.mode == 1:
        clf = tree.DecisionTreeClassifier(random_state=args.seed,max_depth=args.depth)
    else:
        clf = RandomForestClassifier(n_estimators=args.mode,random_state=args.seed,max_depth=args.depth)
    clf.fit(X, y)
    ## dump leafs for conexp-clj ##
    if args.mode == 1:
        dtree = clf.tree_ 
        # get leaf predicates
        dleafs = extract_tree(dtree, X.columns.tolist(), clf.classes_)
        # dump them into a file for conexp-clj. Based on this all scales can be computed
        dump_dleafs(dleafs, args,1)        
    else:
        for i in range(args.mode):
            if not os.path.isdir(f"../output/{args.data}/{args.mode}/{i}"):
                os.makedirs(f"../output/{args.data}/{args.mode}/{i}")
            dtree = clf.__getitem__(i).tree_ 
            # get leaf predicates
            dleafs = extract_tree(dtree, X.columns.tolist(), clf.classes_)
            # dump them into a file for conexp-clj. Based on this all scales can be computed
            dump_dleafs(dleafs, args,i)   
    return clf

################# Evaluation
def dtree_quality(X_train, X_test, y_train, y_test, clf,args): # not yet for all trees of the RF 
    print("Evaluate Tree quality")
    # compute classifier scores 
    acc_train = accuracy_score(y_train,clf.predict(X_train))
    acc_test = accuracy_score(y_test,clf.predict(X_test))
    auc_train = roc_auc_score(y_train,clf.predict_proba(X_train)[:,1])
    auc_test = roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
    # build command for the conexp-clj scale and scaling error  computation
    # use dataframe io for pp
    scores = pd.DataFrame([["Train",acc_train,auc_train]
                           ,["Test",acc_test,auc_test]]
                          ,columns=["type","Accuracy", "AUC"])

    print(scores[["type","Accuracy", "AUC"]].set_index("type"))  
    return scores

def feature_importances(clf,X,y,args):
    entropy = clf.feature_importances_
    permutation = permutation_importance(clf, X, y, n_repeats=10,random_state=42)["importances_mean"]
    I = pd.DataFrame([[X.columns[i],entropy[i],permutation[i]] for i in range(X.shape[1])],
                     columns=["Feature","Entropy","Permutation"])
    I.to_csv(f"../output/{args.data}/{args.mode}/{args.data}_feature_importance.csv",index=False)


def dtree_scalings(X, y, args):
    '''Lets conexp-clj compute all the scalings'''
    thecommand = clojure_cmd(args,"C").split(" ")
    print("Execute Conexp")
    print(thecommand)
    process = Popen(thecommand, stdout=PIPE, stderr=PIPE)
    process.wait()
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)
    if args.mode > 1:
        thecommand = clojure_cmd(args,"RF").split(" ")
        print("Execute Conexp for RF scale")
        print(thecommand)
        process = Popen(thecommand, stdout=PIPE, stderr=PIPE)
        process.wait()
        stdout, stderr = process.communicate()
        print(stdout)
        with open(f"../output/{args.data}/{args.mode}/{args.data}.scales",'w') as s_file:
            print(stdout,file=s_file)
        print(stderr)

def view_concepts(args):
    thecommand = clojure_cmd(args,"CC").split(" ")
    print("Execute Conexp for number of concepts")
    print(thecommand)
    process = Popen(thecommand, stdout=PIPE, stderr=PIPE)
    process.wait()
    stdout, stderr = process.communicate()
    print(stdout)
    with open(f"../output/{args.data}/{args.mode}/{args.data}.concepts",'w') as c_file:
        print(stdout,file=c_file)
    print(stderr)

def clojure_cmd(args, clj_mode):
    '''Compute the scales and the conceptual scaling error in clojure'''
    return "java -jar ../tree2ctx/target/uberjar/tree2ctx-0.2.0-SNAPSHOT-standalone.jar %s -m %s -r %i"%(args.data,clj_mode,args.mode)

################# 

def pipeline(X_train, X_test, y_train, y_test, args): # remove
    """Train the classifier and compute the validation scores"""
    print("Train classifier")
    clf = train_tree(X_train,y_train,args)
    print("Classifier trained")
    thequality = dtree_quality(X_train, X_test, y_train, y_test, clf, args)
    return thequality


def worker(args):
    '''One run of the experiment. Includes:
        - Cross Validation
        - Training of the Decision Tree classifier
        - Scores: Conceptual scaling error(computed by conexp-clj), accuracy, auc and number of scale extents'''
    print("Start Analysis of::: " + args.data)
    try:
        X, y = load_data(args)
        # results of each cross validation step
        result = pd.DataFrame([],columns=["type","Accuracy", "AUC","Scale Error","Tree Exts"])
        # generate CV_NUM random numbers for the cross validation splits
        # set seed for reproducability 
        random.seed(args.seed)
        rs = [random.randint(0,1000) for i in range(CV_NUM)]
        # perform cross validation
        for i in range(CV_NUM):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rs[i])
            # dump used train test split for potential clojure stuff
            dump_transformed_split_data(X_train, X_test, y_train, y_test, args)
            # train classifier and return quality scores
            split_result = pipeline(X_train, X_test, y_train, y_test,args)
            split_result["type"] = split_result["type"].apply(lambda x: "%s_%i"%(x,i))
            # append results of the cv step
            result = pd.concat([result,split_result])
        # dump sclaing on whole data set
        dump_transformed_data(X, y, args)
        clf = train_tree(X,y,args) # todo
        overall_result = dtree_quality(X, X, y, y, clf,args)
        result = pd.concat([result,overall_result])
        # dump overall scores
        result.to_csv(f"../output/{args.data}/{args.mode}/{args.data}_result.csv",index=False)

        feature_importances(clf,X,y,args)
        dtree_scalings(X,y,args) #todo
        view_concepts(args)
    except Exception as e:
        with open(f"../output/{args.data}/{args.mode}/{args.data}.err",'w') as error_file:
            print(sys.exc_info()[0],file=error_file)
            print(e,file=error_file)


if __name__ == "__main__":
    # parse args
    # args is globally available
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data")
    parser.add_argument("-t", "--target", default="target")
    parser.add_argument("-m", "--mode", default=1) # 1 is DT and n is RF size
    parser.add_argument("-n", "--depth", default=None) # 1 is DT and n is RF size
    parser.add_argument("-s", "--seed", default=42)
    args = parser.parse_args()
    args.mode= int(args.mode)
    args.seed= int(args.seed)
    args.depth= None if args.depth == "None" else int(args.depth)
    tree_num = int(args.mode)
    if not os.path.isdir(f"../output/{args.data}"):
        os.makedirs(f"../output/{args.data}")
    if not os.path.isdir(f"../output/{args.data}/{args.mode}"):
        os.makedirs(f"../output/{args.data}/{args.mode}")
    if int(args.mode) > 1:
        if not os.path.isdir(f"../output/{args.data}/{args.mode}/RF"):
            os.makedirs(f"../output/{args.data}/{args.mode}/RF")
    # if args.data == "all":
    #     pool = mp.Pool(min(NUM_CPU,mp.cpu_count()))
    #     pool.map_async(worker, [file[:-4] for file in os.listdir("../../data") if file.endswith(".csv")])
    #     pool.join()
    #     pool.close()
    # else:
    worker(args=args)
