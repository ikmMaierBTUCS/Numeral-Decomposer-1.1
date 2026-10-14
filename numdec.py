'''
Numeral Decomposer 1.1

Decomposes numeral words into functions with inputs, e.g.
'two hundred and twenty-seven' --> '_ hundred and _'(2,27)
'''

import ast
import os
import sys
import unicodedata

import numpy as np
import pandas as pd
from num2words import num2words
from diophantine import solve
from alphabet_detector import AlphabetDetector

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NUMERAL_CSV = os.path.join(DATA_DIR, 'Numeral.csv') # numerals from https://www.languagesandnumbers.com
DVNUM_CSV = os.path.join(DATA_DIR, 'DVnum.csv')

class Vocabulary:
    def __init__(self, nb, nal):
        if isinstance(nb, np.generic):
            nb=nb.item()
        if not type(nal) is str:
            raise TypeError("Numeral has to be a string")
        self.number = nb
        self.word = nal
        self.root = nal
        self.inputrange = []
        self.mapping = [nb]
    def printVoc(self):
        print(str(self.number)+' '+self.word)
    def all_outputs(self):
        return [self]
    def dimension(self):
        return 0
    def actual_dimension(self):
        return 0
    def sample(self):
        return self
class Highlight:
    def __init__(self, voc, start):
        self.number=voc.number
        self.numeral=voc.word
        self.start=start
        self.root=voc.word
        self.mapping=[voc.number]
        self.word=voc.word
    def end(self):
        return self.start+len(self.numeral)
    def hlrange(self):
        return range(self.start+1,self.end())
    def voc(self):
        return Vocabulary(self.number,self.numeral)

class SCFunction:
    '''
    Root is the exponent of the function where _ mark input slots
    Inputrange is a list of lists. The nth lists all SCFunctions that may enter the nth input slot
    Mapping is a list of coefficients. The nth coefficient is the factor by which the nth input would have to be multiplied
        Exception: the last coefficient is the constant coefficient.
    '''
    def __init__(self,root,i,mapping):
        if not type(root) is str:
            raise TypeError("Root has to be a string")
        if not type(i) is list:
            raise TypeError("Inputrange has to be a list")
        dimension=root.count('_')
        if not len(i) == dimension:
            print(i)
            raise TypeError(str(dimension)+"-dimensional function needs "+str(dimension)+" component domains.")
        for component in i:
            if not type(component) is list:
                print(type(component))
                raise TypeError("All component domains have to be lists")
            for entry in component:
                if not type(entry) is Vocabulary and not type(entry) is SCFunction:
                    print(type(entry))
                    raise TypeError("All entries of all input components have to be Vocabulary or SCFunction")
        if not type(mapping) is list or dimension+1!=len(mapping):
            print(type(mapping))
            try:
                print(len(mapping))
            except TypeError:
                pass
            raise TypeError("Mapping has to be a list of "+str(dimension)+"+1 coefficients")
        self.root=root
        self.inputrange=i
        self.mapping=mapping
    def dimension(self):
        '''
        Dimension = Number of input slots
        '''
        return self.root.count('_')
    def number_inputs(self):
        ni = []
        for comp in self.inputrange:
            ni += [[entr.sample().mapping[-1] for entr in comp]]
        return ni
    def input_numberbase(self):
        build_base=[]
        base_complete = False
        for root_inputx in cartesian_product(self.inputrange):
            for final_inputx in cartesian_product([entr.all_outputs() for entr in root_inputx]):
                # if inputx is not in span(build_base), i.e. matrank(build_base+inputx)=len(build_base)+1
                if np.linalg.matrix_rank(np.array(build_base+[[component.mapping[-1] for component in final_inputx]+[1]], dtype=np.float64),tol=None)==len(build_base)+1:
                    build_base=build_base+[[component.mapping[-1] for  component in final_inputx]+[1]]
                if len(build_base)==self.dimension()+1:
                    base_complete = True
                    break
            if base_complete:
                break
        return build_base

    def actual_dimension(self):
        '''
        Dimension of input range with respect to affine linearity
        '''
        numbers = [[self.inputrange[i][j].all_outputs()[0].mapping[-1] for j in range(len(self.inputrange[i]))] for i in range(self.dimension())]
        return np.linalg.matrix_rank(np.array(cartesian_product(numbers+[[1]]), dtype=np.float64))
    def insert(self,inputx):
        '''
        Requires a dimension-long list of input SCFunctions
        Return a new SCFunction where all input SCFunctions are inserted in their respective slot
        Updates inputrange and mapping with respect to new inputslots originating from the input SCFunctions
        '''

        # catch errors
        if not type(inputx) is list:
            print(type(inputx))
            raise TypeError('Input has to be a list')
        if len(self.inputrange) != len(inputx) or len(self.inputrange) == 0:
            raise TypeError("Input does not match dimension or "+self.root+" has no inputslots.")
        for component in inputx:
            if not type(component) is Vocabulary and not type(component) is SCFunction:
                print(type(component))
                raise TypeError('All components of the input have to be a Vocabulary or an SCFunction')

        # initialize root, inputrange and mapping of composed SCF
        rootparts = self.root.split('_')
        output_root = ''
        output_i = []
        output_mapping = []
        constant_coefficient = self.mapping[-1]

        # extend root, inputrange and mapping
        for inp in range(len(inputx)):
            output_root += rootparts[inp] + inputx[inp].root
            output_i += inputx[inp].inputrange
            output_mapping += [self.mapping[inp] * coeff for coeff in inputx[inp].mapping[:-1]]
            constant_coefficient += self.mapping[inp] * inputx[inp].mapping[-1]

        # finish root and mapping and return composed SCF
        output_root += rootparts[-1]
        output_mapping += [constant_coefficient]
        return SCFunction(output_root,output_i,output_mapping)
    def sample(self):
        if self.dimension() == 0:
            return self
        else:
            return self.insert([comp[0].sample() for comp in self.inputrange])

    def all_outputs(self):
        '''
        return all final SCFunctions (vocabulary) without unsatisfied '_'s left, that are derivable from
        '''
        if self.dimension() == 0:
            return [self]
        else:
            all_output = []
            for inputvector in cartesian_product(self.inputrange):
                new_outputs = self.insert(inputvector).all_outputs()
                all_output += new_outputs
            return all_output
    def all_outputs_as_voc(self):
        ao = self.all_outputs()
        aov = []
        for scf in ao:
            aov += [Vocabulary(scf.mapping[-1],scf.root)]
        return aov

    def merge(self,mergee):
        if not type(mergee) is SCFunction:
            print('not scf')
            raise TypeError("Can only merge with other SCFunction")
        if not self.root==mergee.root:
            print('different roots')
            raise BaseException('Cannot merge with SCFunction with different exponent')
        if any(len(mergee.inputrange[comp])>1 for comp in range(self.dimension())):
            print('mergee is not singleton')
            raise BaseException('Merge of SCFunctions is yet only implemented for mergees directly produced by proto_parse')
        if self.dimension() == 0:
            print('merger is not generalizable')
            raise BaseException('SCFunction of dimension 0 cannot merge')
        new_inputrange = []
        for comp in range(self.dimension()):
            if mergee.inputrange[comp][0].root in [ent.root for ent in self.inputrange[comp]]:
                new_inputrange += [self.inputrange[comp]]
            else:
                new_inputrange += [self.inputrange[comp] + mergee.inputrange[comp]]
        insert=self.insert([mergee.inputrange[comp][0] for comp in range(mergee.dimension())])
        if insert.mapping[-1] == mergee.mapping[-1]:
            # current mapping predicts value of the mergee correctly
            return SCFunction(self.root,new_inputrange,self.mapping)
        else:
            # The current mapping does not fit the mergee. The mergee can only be merged if it raises the
            # dimension of the input range, so that a new mapping can be solved. Otherwise it needs a separate function.
            build_base = [b for b in self.input_numberbase()]
            build_image = [np.dot(self.mapping,basevec) for basevec in build_base]
            new_dim = np.linalg.matrix_rank(np.array([[component[0].number for component in mergee.inputrange]+[1]] + build_base, dtype=np.float64))
            if new_dim > self.actual_dimension():
                # expand base and image
                build_base = [[component[0].number for component in mergee.inputrange]+[1]] + build_base
                build_image = [mergee.mapping[-1]] + build_image
            else:
                self.sample().present()
                print('No merge')
                self.present()
                mergee.present()
                raise BaseException('Mergee must have a different mapping')
            coefficients=intlinsolve([b[:-1] for b in build_base],build_image)
            return SCFunction(self.root,new_inputrange,list(coefficients))
    def present(self, domain=True, printout=True):
        if self.dimension() == 0:
            if printout:
                print(self.root+" is "+str(self.mapping[-1]))
            return self.root+" is "+str(self.mapping[-1])
        else:
            domainstrs = []
            for comp in self.inputrange:
                component = '{'
                for entry in comp:
                    if isinstance(entry,Vocabulary) or entry.dimension() == 0:
                        component += str(entry.mapping[-1])+','
                    else:
                        component += str(entry.root)+','
                component = component[:-1]+'}'
                domainstrs += [component]
            domainstr = 'x'.join(domainstrs)
            if self.dimension() == 1:
                inpstr = 'x'
                outpstr = str(self.mapping[0]) + 'x+' + str(self.mapping[1])
            else:
                variables = ['x','y','z','a','b','c','d','e','f','g','h']
                inpstr = '('
                outpstr = ''
                for comp in range(self.dimension()):
                    inpstr += variables[comp]+','
                    outpstr += str(self.mapping[comp]) + variables[comp] + '+'
                inpstr = inpstr[:-1] + ')'
                if self.mapping[-1] != 0:
                    outpstr += str(self.mapping[-1])
                else:
                    outpstr = outpstr[:-1]
            if domain:
                retstr = "Function " + self.root + "\t maps " + domainstr + "\t by " + inpstr + '\t -> ' + outpstr
            else:
                retstr = self.root + "\t maps " + inpstr + '\t -> ' + outpstr
            if printout:
                print(retstr)
            return retstr
    def reinforce(self,lexicon,supervisor,printb=True):
        copy = self
        candidates_for_abstraction = []
        upper_limit = sum([max([0,coeff]) for coeff in self.mapping])
        for entry in lexicon:
            if all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs()]):
                candidates_for_abstraction += [entry]
        invariant_slots = []
        for comp in range(copy.dimension()):
            if len(copy.inputrange[comp]) == 1:
                invariant_slots += [comp]
        for combination in cartesian_product((copy.dimension() - len(invariant_slots)) * [candidates_for_abstraction]):
            cand = []
            for comp in range(copy.dimension()):
                if comp in invariant_slots:
                    cand += [copy.inputrange[comp][0]]
                else:
                    cand += [combination[0]]
                    combination = combination[1:]
            word_cand = copy.insert([candc.sample() for candc in cand])
            proposed_inputrange = []
            entry_is_new = False
            for comp in range(copy.dimension()):
                if cand[comp].root in [entr.root for entr in copy.inputrange[comp]]:
                    proposed_inputrange += [copy.inputrange[comp]]
                else:
                    proposed_inputrange += [copy.inputrange[comp]+[cand[comp]]]
                    entry_is_new = True
            if entry_is_new:
                proposal = SCFunction(copy.root,proposed_inputrange,copy.mapping)
                proposal_is_new = True
                for entr in lexicon:
                    for word in entr.all_outputs():
                        if word_cand.root == word.root or word_cand.mapping[-1] == word.mapping[-1]:
                            # proposal is already covered by the lexicon
                            proposal_is_new = False
                            break
                    if not proposal_is_new:
                        break
                if proposal_is_new and proposal.actual_dimension() == copy.actual_dimension():
                    # ask the supervisor: can I also say word_cand?
                    for v in supervisor:
                        if word_cand.root == v.word:
                            cand_parse = advanced_parse(word_cand.mapping[-1],word_cand.root,lexicon,False,False)
                            if cand_parse.root == self.root:
                                if v.number != word_cand.mapping[-1]:
                                    print('LEARNING ERROR: ' + v.word + ' is ' + str(v.number) + ' but learner assumes ' + str(word_cand.mapping[-1]))
                                copy = proposal
                            else:
                                if printb: print('OK, but I think this is not related')
                            break
        return copy

def intlinsolve(base,image):
    try:
        solution = list(solve(base,image)[0])
        return solution+[0]
    except IndexError:
        # constant needed
        try:
            return solve([b+[1] for b in base],image)[0]
        except NotImplementedError:
            # unique solution
            return [round(i) for i in np.dot(np.linalg.pinv(np.array([b+[1] for b in base], dtype=np.float64)),image)]
    except NotImplementedError:
        # unique solution
        return [round(i) for i in np.dot(np.linalg.pinv(np.array(base, dtype=np.float64)),image)]+[0]

def cartesian_product(listlist):
    if len(listlist)==0:
        print('ERROR: empty input in product')
        return False
    cp=[[x] for x in listlist[0]]
    for liste in listlist[1:]:
        cp=[a+[b] for a in cp for b in liste]
    return cp


def delatinized(string):
    '''
    Removes the latin transcription from a numeral that is given in two scripts
    '''
    ad = AlphabetDetector()
    if not ad.is_latin(string):
        if not ad.is_cyrillic(string):
            if ad.is_cyrillic(string[0]) and not ad.is_cyrillic(string[-1]):
                # first part is cyrillic
                for point in range(len(string)):
                    if not ad.is_cyrillic(string[:point]):
                        return string[:point-2]
            elif not ad.is_cyrillic(string[0]) and ad.is_cyrillic(string[-1]):
                # last part is cyrillic
                for point in reversed(range(len(string))):
                    if not ad.is_cyrillic(string[point:]):
                        return string[point+2:]
            elif ad.is_latin(string[0]):
                # first part is latin
                for point in range(len(string)+1):
                    if not ad.is_latin(string[:point]):
                        return string[point-1:]
            elif ad.is_latin(string[-1]):
                # last part is latin
                for point in reversed(range(len(string)+1)):
                    if not ad.is_latin(string[point:]):
                        return string[:point+1]
            else:
                return string
        else:
            return string
    else:
        return string

def create_lexicon(language,set_limit=10**9):
    '''
    Returns a list of Vocabulary for the language, taken from num2words, Numeral.csv or DVnum.csv
    '''
    LEX=[]
    try:
        num2words(1, lang=language)
        for integer in list(range(1,1001))+[1002,1006,1100,1200,1206,7000,7002,7006,7100,7200,7206,10000,17000,17200,17206,20000,27000,27006,27200,27206]:
            if integer < set_limit:
                try:
                    numeral=num2words(integer, lang=language)
                    voc=Vocabulary(integer,numeral)
                    LEX=LEX+[voc]
                except:
                    pass
        return LEX
    except NotImplementedError:
        try:
            lanu=pd.read_csv(NUMERAL_CSV, encoding = "utf_16", sep = '\t')
            df=lanu[lanu['Language']==language]
            biscriptual=False
            if ' ' in df.iloc[0,2]:
                biscriptual=True
            for i in range(len(df)):
                if i+1 < set_limit:
                    numeral=df.iloc[i,2]
                    if numeral[0]==' ':
                        numeral=numeral[1:]
                    if numeral[-1]==' ':
                        numeral=numeral[:-1]
                    if language in ['Latin','Persian','Arabic']:
                        words=numeral.split(' ')
                        numeral=' '.join(iter(words[:-1]))
                    if language in ['Chuvash','Adyghe']:
                        words=numeral.split(' ')
                        numeral=' '.join(iter(words[:len(words)//2]))
                    if biscriptual and not language in ['Chuvash','Adyghe','Latin','Persian','Arabic']:
                        numeral=delatinized(numeral)
                    numeral=numeral.replace('%',',')
                    voc=Vocabulary(i+1,numeral)
                    LEX=LEX+[voc]
            return LEX
        except IndexError:
            lanu=pd.read_csv(DVNUM_CSV, encoding = "utf_8", sep = ';')
            df=lanu[lanu['Language']==language]
            for i in range(len(df)):
                if i+1 < set_limit:
                    numeral=df.iloc[i,2]
                    if numeral[0]==' ':
                        numeral=numeral[1:]
                    if numeral[-1]==' ':
                        numeral=numeral[:-1]
                    voc=Vocabulary(i+1,unicodedata.normalize('NFC',numeral))
                    LEX=LEX+[voc]
            if len(LEX)==0:
                raise NotImplementedError("Language "+language+" is not supported or spelled differently")
            return LEX

def vocabulary_of(lexicon):
    '''
    Returns the lexicon as a list of Vocabulary, i.e. SCFunctions in the lexicon are replaced by all their outputs
    '''
    if len(lexicon) != 0 and isinstance(lexicon[0],Vocabulary):
        return lexicon
    vocabulary = []
    for entry in lexicon:
        if isinstance(entry,Vocabulary):
            vocabulary += [entry]
        else:
            vocabulary += entry.all_outputs_as_voc()
    return vocabulary

def proto_parse(number,numeral,lexicon,print_documentation,print_result): #parse a (int number,str numeral)-pair using (current) lexicon 'lexicon'. boolean print_documentation toggles documentation printout. boolean print_result toggles result printout
    if print_documentation: print('parse '+numeral+' '+str(number))
    lex1=vocabulary_of(lexicon)+[Vocabulary(number,numeral)] # so the new word itself is found at the end
    checkpoint=0 # point from which parsing is finally performed already
    highlights=[] #list of highlights, initially empty
    for end in range(len(numeral)+1): #set end of the observed substring
        startrange=range(checkpoint,end) #start of observed string may lie between checkpoint and end
        for highlight in highlights:
            startrange=set(startrange)-set(highlight.hlrange()) # observed strings may not start inside present highlights. rather they have to fully contain a highlight or be disjoint with it
        startrange=sorted(list(startrange)) # so ints in startrange are sorted by size
        for start in startrange: # set start of observed substring of numeral
            subnum_found_at_this_end=False # boolean condition to break start-loop
            substring=numeral[start:end] #set observed substring
            if print_documentation: print('substring: ',substring)
            for entry in lex1: #browse current lexicon
                if entry.word==substring: #look if substring appears
                    subnum_found_at_this_end=True
                    if 2*entry.number<number: #highlighting condition
                        if print_documentation: print(substring+' <' + str(entry.number) + '/2')
                        for highlight in highlights[:]: #browse through present highlights
                            if highlight.start>=start: # if a highlight is contained in new highlight,...
                                if print_documentation: print("remove "+highlight.numeral)
                                highlights.remove(highlight) #then remove it from list of highlights
                        highlights=highlights+[Highlight(entry,start)] # add new highlight
                        if print_documentation: print('Unpacked: ['+','.join([str(highlight.numeral) for highlight in highlights])+']')
                    else:
                        if print_documentation: print(substring+' >=' + str(entry.number) + '/2')
                        checkpoint=end
                        if print_documentation: print("Set checkpoint behind "+numeral[:checkpoint])
                    break # out of browsing the lexicon
            if subnum_found_at_this_end:
                break # out of start-loop
    root=numeral
    for highlight in reversed(highlights):
        root=root[0:highlight.start]+'_'+root[highlight.end():len(root)]
    decompstr=str(number)+'='+root+'('
    decompstr=decompstr+','.join([str(highlight.number) for highlight in highlights])
    decompstr=decompstr+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral)] for highlight in highlights],[0 for highlight in highlights]+[number])

def advanced_parse(number, word, lexicon, print_doc, print_result):
    if print_doc: print('parse '+word+' '+str(number))
    lexicon1 = vocabulary_of(lexicon)+[Vocabulary(number,word)]
    checkpoint = 0
    highlights=[]
    mult_found=False
    for end in range(0, len(word)+1):
        startrange=set(range(checkpoint, end))
        for highlight in highlights:
            startrange=startrange-set(highlight.hlrange())
        startrange=sorted(list(startrange))
        for start in startrange:
            subnum_found_at_this_end=False
            substr=word[start:end]
            if print_doc: print('substring = '+str(substr))
            for entry in lexicon1:
                if substr == entry.word:
                    subnum_found_at_this_end=True
                    subentry_found = False
                    if 2*entry.number < number or mult_found:
                        if print_doc: print(substr+' is <' + str(number) + '/2')
                        for highlight in reversed(highlights):
                            if highlight.start >= start:
                                if print_doc: print("remove "+highlight.numeral)
                                highlights.remove(highlight)
                        highlights=highlights+[Highlight(Vocabulary(entry.number,entry.word),start)]
                        if print_doc: print('Unpacked: ',[highlight.numeral for highlight in highlights])
                    else:
                        if print_doc: print(substr+" is >=" + str(number) + '/2')
                        mult_found=True
                        checkpoint=end
                        earliest_laterstart = start+1
                        for highlight in highlights:
                            if highlight.number**2 < entry.number:
                                earliest_laterstart = min(end,highlight.end()) #so factors remain untouched
                        potential_highlight = None
                        for laterstart in range(earliest_laterstart,end):
                            if print_doc: print('subnum = '+word[laterstart:end])
                            for subentry in lexicon1:
                                if word[laterstart:end] == subentry.word:
                                    if subentry.number**2 <= entry.number:
                                        # subnumeral is FAC or SUM. If it would contain MULT, its square would be larger than entry
                                        if print_doc:
                                            print(word[laterstart:end]+" is <sqrt("+str(entry.number)+")")
                                            if potential_highlight:
                                                print("Ignore " + potential_highlight.word + " because " + word[laterstart:end] + " is its subnumeral.")
                                        subentry_found=True
                                        potential_highlight = Highlight(Vocabulary(subentry.number,subentry.word),laterstart)
                                        potential_checkpoint = laterstart
                                        if print_doc: print('Unpacked: ',[highlight.numeral for highlight in highlights])
                                    else:
                                        if entry.number % subentry.number != 0 and 2*subentry.number < number:
                                            # Subnumeral probably contains SUM. As it is no divisor of entry, entry has to contain SUM.
                                            # It cannot contain FAC*MULT, as it is smaller than half of the numeral. And it cannot be FAC,
                                            # as its square is larger than entry. So it is composed of SUM and MULT.
                                            # If it turns out to be irreducible with the present properties, we assume it is SUM
                                            if print_doc:
                                                print(word[laterstart:end]+" is at least <" + str(number) + "/2 and is no divisor of " + entry.word + ".")
                                                if potential_highlight:
                                                    print("Ignore " + potential_highlight.word + " because " + word[laterstart:end] + " is its subnumeral.")
                                            potential_highlight = Highlight(Vocabulary(subentry.number,subentry.word),laterstart)
                                            potential_checkpoint = laterstart
                                        elif entry.number % subentry.number == 0:
                                            if print_doc: print(str(subentry.number) + " is divisor of " + str(entry.number) + " and is >=sqrt(" + str(entry.number) + ").")
                                            potential_highlight = None
                                        elif 2*subentry.number >= number:
                                            if print_doc: print(str(subentry.number) +" is at least <" + str(number) + "/2")
                                            potential_highlight = None
                            if subentry_found:
                                break
                        if potential_highlight != None:
                            for highlight in reversed(highlights):
                                if highlight.end() > potential_checkpoint:
                                    if print_doc: print("remove "+highlight.root)
                                    highlights.remove(highlight)
                            highlights = highlights+[potential_highlight]
                            checkpoint = potential_checkpoint
                            if print_doc: print('Unpacked: ',[highlight.root for highlight in highlights])
                        if print_doc: print("Set checkpoint behind "+word[:checkpoint])
                    break
            if subnum_found_at_this_end:
                break
    if len(highlights) == 2:
        if highlights[0].number + highlights[1].number == number:
            sohi = sorted(highlights, key=lambda highlight: highlight.number)
            suspected_mult = sohi[-1]
            if print_doc: print("remove "+suspected_mult.root+' because ' + sohi[0].root + ' + ' + sohi[1].root + " = " + word + " and "+ sohi[0].root + ' > ' + sohi[1].root + "so it is probably mult.")
            highlights.remove(suspected_mult)
        elif highlights[0].number * highlights[1].number == number:
            sohi = sorted(highlights, key=lambda highlight: highlight.number)
            suspected_mult = sohi[-1]
            if print_doc: print("remove "+suspected_mult.root+' because ' + sohi[0].root + ' * ' + sohi[1].root + " = " + word + " and "+ sohi[0].root + ' > ' + sohi[1].root + "so it is probably mult.")
            highlights.remove(suspected_mult)
    elif len(highlights) == 3:
        sohi = sorted(highlights, key=lambda highlight: highlight.number)
        suspected_mult = sohi[-1]
        if suspected_mult.number**2 > number:
            other_numbers = [highlight for highlight in highlights if highlight != suspected_mult]
            if other_numbers[0].number * suspected_mult.number + other_numbers[1].number == number:
                if print_doc: print("remove "+suspected_mult.root+ " because " + other_numbers[0].root + " * " + suspected_mult.root + " + " + other_numbers[1].root + " = " + word + " and " + suspected_mult.root + " > " + other_numbers[0].root + " so it is probably mult.")
                highlights.remove(suspected_mult)
            elif other_numbers[1].number * suspected_mult.number + other_numbers[0].number == number:
                if print_doc: print("remove "+suspected_mult.root+ " because " + other_numbers[1].root + " * " + suspected_mult.root + " + " + other_numbers[0].root + " = " + word + " and " + suspected_mult.root + " > " + other_numbers[1].root + " so it is probably mult.")
                highlights.remove(suspected_mult)
    elif len(highlights) > 3:
        suspected_mult = max(highlights, key=lambda highlight: highlight.number)
        if suspected_mult.number**2 > number:
            other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            for suspected_factor in other_numbers:
                suspected_summand = sum(other_numbers)-suspected_factor
                if suspected_factor * suspected_mult.number + suspected_summand == number:
                    if print_doc: print("remove "+suspected_mult.root+' since it is probably mult.')
                    highlights.remove(suspected_mult)
                    break

    root=word
    for highlight in reversed(highlights):
        root=root[0:highlight.start]+'_'+root[highlight.end():len(root)]
    decompstr = str(number)+'='+root+'('+','.join([str(highlight.number) for highlight in highlights])+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral)] for highlight in highlights],[0 for highlight in highlights]+[number])


def list_scfunctions(language,printb=True,set_limit=999999,version='new'):
    '''
    Decomposes all numerals of the language, merges decompositions with the same root into one SCFunction
    and returns the lexicon size, i.e. the number of SCFunctions plus the number of irreducible numerals.
    version='new' uses advanced_parse, version='old' uses proto_parse
    '''
    if version == 'new':
        parse_numeral = advanced_parse
        decomposer = 'advanced'
    elif version == 'old':
        parse_numeral = proto_parse
        decomposer = 'prototype'
    if printb: print('Parse numerals in '+language)
    lex=[entry for entry in create_lexicon(language) if entry.number<set_limit]
    limit = len(lex)
    if limit > 1000:
        limit = 999999
    set_of_scfunctions=[]
    irreducibles=[]
    for entry in lex:
        parse=parse_numeral(entry.number,entry.word,lex,False,False)
        if '_' in parse.root:
            function_known=False
            for pos in range(len(set_of_scfunctions)):
                if set_of_scfunctions[pos].root == parse.root:
                    set_of_scfunctions[pos]=set_of_scfunctions[pos].merge(parse)
                    function_known=True
                    break
            if not function_known:
                set_of_scfunctions=set_of_scfunctions+[parse]
        else:
            irreducibles += [parse]
    if printb: print(language+' has '+str(len(set_of_scfunctions))+' numeral functions and '+str(len(irreducibles))+' irreducible numerals when using the '+decomposer+' numeral decomposer for numerals till '+ str(limit) +'.')
    if printb: print('The number functions are:')
    printout=""
    for scf in set_of_scfunctions:
        example=scf.sample()
        printout=printout + scf.present(printout=False,domain=True)+', \t e.g. '+str(example.mapping[-1])+' is '+str(example.root)+'\n'
    if printb: print(printout)
    if printb: print('The irreducibles are: ' + ', '.join([scf.root+' ('+str(scf.mapping[-1])+')' for scf in irreducibles]))
    if printb: print('')
    return len(set_of_scfunctions) + len(irreducibles)

def decompose_numeral(number,language,version='new'):
    lex = create_lexicon(language)
    for entr in lex:
        if entr.number == number:
            word = entr.word
            break
    else:
        raise ValueError("The lexicon of "+language+" has no numeral for "+str(number))
    decompose_numeral_custom(number,word,lex,version)

def decompose_numeral_custom(number,word,lexicon,version='new'):
    if isinstance(lexicon,str):
        try:
            lex = create_lexicon(lexicon)
        except NotImplementedError:
            raise NotImplementedError("Assuming " + lexicon + " is a language, it is not supported. Use a custom lexicon to test the decomposer on it")
    elif isinstance(lexicon,list):
        if len(lexicon)==0:
            lex = lexicon
        elif isinstance(lexicon[0],list):
            try:
                lex = [Vocabulary(l[0],l[1]) for l in lexicon]
            except (IndexError, TypeError):
                raise TypeError("Custom lexicon must be a list of Vocabulary objects or of (number,numeral) pairs, e.g. [[1,'one'],[2,'two']], ")
        elif isinstance(lexicon[0],Vocabulary):
            lex = lexicon
        else:
            raise TypeError("Custom lexicon must be a list of Vocabulary objects or of (number,numeral) pairs, e.g. [[1,'one'],[2,'two']], ")

    else:
        raise TypeError("Lexicon must be a string representing a language or a list serving as a custom lexicon")
    if version == 'new':
        advanced_parse(number,word,lex,True,True)
    elif version == 'old':
        proto_parse(number,word,lex,True,True)


def main():
    sys.stdout.reconfigure(errors='backslashreplace') # numerals the terminal cannot encode are printed as \u-escapes instead of crashing
    print("Welcome to Numeral Decomposer")
    print("a: Advanced decomposer")
    print("p: Prototype decomposer")
    version = str(input("Choose a version by typing 'a' or 'p': "))

    if version == 'a':
        print("You chose the advanced decomposer to decompose a numeral")
        version = 'new'
    elif version == 'p':
        print("You chose the prototype decomposer to decompose a numeral")
        version = 'old'
    else:
        raise ValueError("Invalid choice")

    numeral = str(input("Type in the numeral word to decompose: "))

    number = int(input("Type in the number value of the numeral: "))

    print("The decomposer needs a lexicon to work with. You can use a predefined lexicon of numerals from a natural language.")
    print("Or you customize a lexicon to test unsupported or invented (variations of) languages")
    customize = str(input("Do you want to customize the lexicon? [y/n]: "))

    if customize == 'n':
        try:
            language = str(input("Type in the language of your numeral: "))
            lexicon = create_lexicon(language)
        except NotImplementedError:
            print("We have no predefined lexicon for a language spelled " + language + ".")
            print("Try again by spelling the language exactly as in https://github.com/ikmMaierBTUCS/Numeral-Decomposer-1.1/blob/main/languages.txt")
            print("If your language has no predefined lexicon, you could customize a lexicon instead.")
            language = str(input("Type in the language of your numeral: "))
            lexicon = create_lexicon(language)
        if number > 1000:
            print("Warning: The predefined lexicon may not cover numbers over 1000. In order to assure a proper decomposition, add your numeral's subnumerals larger than 1000 to the lexicon.")
            print("E.g. when decomposing 'twenty-one thousand and two', add '[[21000,'twenty-one thousand'],[1000,'one thousand'],[1002,'one thousand and two']]'")
            extralexicon = ast.literal_eval(input("Type in the additional lexicon entries or type '[]' to skip: "))
            lexicon += [Vocabulary(l[0],l[1]) for l in extralexicon]
    elif customize == 'y':
        print("Provide a lexicon for the decomposition of your numeral. It should include all subnumerals of your numeral. For each subnumeral that you do not include, you are making a decision that the decomposer is supposed to take.")
        print("Example: In order to decompose enoderdnuhowt (201) in backward English, you may want to provide [[1,'eno'],[100,'derdnuh'],[2,'owt'],[101,'enoderdnuh'],[200,'derdnuhowt']]")
        print("Example: In order to decompose neetxisderdnuhowt (216) in backward English, you may want to provide [[16,'neetxis'],[116,'neetxisderdnuh'],[6,'xis'],[106,'xisderdnuh'],[206,'xisderdnuhowt'],[100,'derdnuh'],[200,'derdnuhowt'],[2,'owt']].")
        lexicon = ast.literal_eval(input("Type in the lexicon of your numeral: "))
        lexicon = [Vocabulary(l[0],l[1]) for l in lexicon]
    else:
        raise ValueError("Invalid choice")

    decompose_numeral_custom(number,numeral,lexicon,version)


if __name__ == "__main__":
    main()
