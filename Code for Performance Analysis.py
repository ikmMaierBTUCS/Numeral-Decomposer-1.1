# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:14:35 2023

@author: ikm
"""

langnumb=pd.read_csv(FILE_PATH_OF_LANGUAGESANDNUMBERS_DATA_CSV, encoding = "utf_16", sep = '\t')
package_languagesandnumbers=set([langnumb.iloc[i,0] for i in range(len(langnumb))])
package_languagesandnumbers.remove('Malecite-Passamaquoddy')
package_languagesandnumbers.remove('Language')
package_num2words=['fr','fi','fr_CH','fr_BE','fr_DZ','he','id','it','ja','kn','ko','lt','lv','no','pl','pt','pt_BR','sl','sr','ro','ru','tr','th','vi','nl','uk','es_CO','es','es_VE','cz','de','ar','dk','en_GB','en_IN']

  '''COMPARE OLD WITH NEW PARSER'''
import matplotlib.pyplot as plt
lexsizes=[]
oldlexsizes=[]
for language in list(package_languagesandnumbers)+package_num2words:
    try:
        lexsize = list_scfunctions(language)
        oldlexsize = old_list_scfunctions(language)
        lexsizes += [lexsize]
        oldlexsizes += [oldlexsize]
    except:
        pass
ratios = [lexsizes[i]/oldlexsizes[i] for i in range(len(lexsizes))]