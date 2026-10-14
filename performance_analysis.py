'''
Compares the lexicon sizes of the advanced (1.1) and the prototype (1.0) numeral decomposer for all languages.
The printout is saved in results/performance_analysis.txt, the plots in results/
'''

import os

import matplotlib.pyplot as plt
import pandas as pd

from numdec import list_scfunctions, NUMERAL_CSV

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

langnumb=pd.read_csv(NUMERAL_CSV, encoding = "utf_16", sep = '\t')
package_languagesandnumbers=set(langnumb['Language'])
package_languagesandnumbers.remove('Malecite-Passamaquoddy')
package_languagesandnumbers.remove('Language')
package_num2words=['fr','fi','fr_CH','fr_BE','fr_DZ','he','id','it','ja','kn','ko','lt','lv','no','pl','pt','pt_BR','sl','sr','ro','ru','tr','th','vi','nl','uk','es_CO','es','es_VE','cs','de','ar','da','en_GB','en_IN']

# compare old with new parser
lexsizes=[]
oldlexsizes=[]
for language in sorted(package_languagesandnumbers)+package_num2words:
    try:
        lexsize = list_scfunctions(language,version='new')
        oldlexsize = list_scfunctions(language,version='old')
        lexsizes += [lexsize]
        oldlexsizes += [oldlexsize]
    except Exception:
        pass
ratios = [lexsizes[i]/oldlexsizes[i] for i in range(len(lexsizes))]

# for the sake of better presentation, the 2 largest values of each version are not plotted
plt.plot(sorted(lexsizes)[:-2], color='blue', label='Numeral Decomposer 1.1')
plt.plot(sorted(oldlexsizes)[:-2], color='green', label='Numeral Decomposer 1.0')
plt.ylabel('lexicon size')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'lexicon_sizes.png'))
plt.close()

plt.plot(sorted(ratios))
plt.ylabel('lexicon size 1.1 / lexicon size 1.0')
plt.savefig(os.path.join(RESULTS_DIR, 'lexicon_size_ratios.png'))
plt.close()
