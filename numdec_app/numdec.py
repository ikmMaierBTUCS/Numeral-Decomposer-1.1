import numpy as np
from num2words import num2words
import itertools
from itertools import combinations
from itertools import permutations
from sympy import Matrix
from diophantine import solve
import pandas as pd
from alphabet_detector import AlphabetDetector

class Vocabulary:
    def __init__(self, nb, nal):
        if type(nb) is list or type(nb) is np.array:
            try:
                nb=nb.item()
            except:
                pass
               # print(type(nb))
                #try:
                    #print(type(nb.item()))
                #except:
                    #pass
                #raise TypeError("Number has to be an integer")
        if not type(nal) is str:
            raise TypeError("Numeral has to be a string")
        self.number = nb
        self.word = nal
        self.root = nal
        self.inputrange = []
        self.mapping = [nb]
    def printVoc(self):
        print(str(self.number)+' '+self.numeral)
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
            except:
                pass
            raise TypeError("Mapping has to be a list of "+str(dimension)+"+1 coefficients")
        #for coeff in mapping:
            #if not type(coeff) is int and not type(coeff) is float:
                #print(type(coeff))
                #raise TypeError("All coefficients of the mapping have to be integers or floats")
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
        #print(len(self.inputrange))
        base_complete = False
        for root_inputx in cartesian_product(self.inputrange):
            for final_inputx in cartesian_product([entr.all_outputs() for entr in root_inputx]):
                #print(build_base)
                #print([component.number for component in final_inputx]+[1])
                #if not inputx in span(build_base): # matrank(buildbase+inputx)=len(Buildbase)+1
                if np.linalg.matrix_rank(np.array(build_base+[[component.mapping[-1] for component in final_inputx]+[1]], dtype=np.float64),tol=None)==len(build_base)+1:
                    #print(str(self.insert(inputx).number)+' is linear independent')
                    build_base=build_base+[[component.mapping[-1] for  component in final_inputx]+[1]]
                    #print([self.insert(inputx).number])
                if len(build_base)==self.dimension()+1:
                    #print('base complete')
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
                
        # trouble shoot if input is not in the inputrange
        for entry in range(len(inputx)):
            if inputx[entry].root not in [inputfunction.root for inputfunction in self.inputrange[entry]]:
                #print("Input "+str([inp.root for inp in inputx])+" is not in the input range of "+str(self.root))
                break
                
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
        #print('alloutputs of '+self.root)
        if self.dimension() == 0:
            return [self]
        else:
            all_output = []
            #print(self.inputrange)
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
        #if mergee.insert(mergee.inputrange[0]).number!=self.insert(mergee.inputrange[0]).number:
            #print('EXPERIMENTAL ERROR! The constructed SCFunction is not affine linear')
        #if [component.number for component in mergee.inputrange[0]] in SPAN(self.inputrange): # insert(Mergee)=mergee.number
        new_inputrange = []
        for comp in range(self.dimension()):
            if mergee.inputrange[comp][0].root in [ent.root for ent in self.inputrange[comp]]:
                new_inputrange += [self.inputrange[comp]]
            else:
                new_inputrange += [self.inputrange[comp] + mergee.inputrange[comp]]
        insert=self.insert([mergee.inputrange[comp][0] for comp in range(mergee.dimension())])
        if insert.mapping[-1] == mergee.mapping[-1]:
            #print('Current mapping predicts value of '+insert.root+' correctly')
            return SCFunction(self.root,new_inputrange,self.mapping)
        else:
            # DANN PRÜFE ERST OB MERGEE NICHT IM SPANN DER INPUTRANGE LIEGT. WENN DOCH, DANN BRAUCHT ES EINE SEPARATE FUNKTION
            # MACH PROPOSAL UND DANN PRÜFE OB ES EINE HÖHERE DIMENSION HAT ALS SELF. WENN NICHT KANN ES NICHT MERGEN
            build_base = [b for b in self.input_numberbase()]
            build_image = [np.dot(self.mapping,basevec) for basevec in build_base]
            print([[component[0].number for component in mergee.inputrange]+[1]] + build_base)
            new_dim = np.linalg.matrix_rank(np.array([[component[0].number for component in mergee.inputrange]+[1]] + build_base, dtype=np.float64))
            #print('newdim determined')
            if new_dim > self.actual_dimension():
                #print('expand base and image')
                build_base = [[component[0].number for component in mergee.inputrange]+[1]] + build_base
                build_image = [mergee.mapping[-1]] + build_image
            else:
                self.sample().present()
                print('No merge')
                self.present()
                mergee.present()
                #print(build_base)
                #print([component[0].number for component in mergee.inputrange]+[1])
                raise BaseException('Mergee must have a different mapping')
            coefficients=intlinsolve([b[:-1] for b in build_base],build_image)
            #try:
                #coefficients=solve(build_base,build_image)[0]
            #except:
                #coefficients=np.dot(np.linalg.pinv(np.array(build_base, dtype=np.float64),rcond=1e-15),build_image)
            #coefficients=[round(coeff) for coeff in coefficients]
            #print(coefficients)
            #print(len(mergee.inputrange+self.inputrange))
            return SCFunction(self.root,new_inputrange,list(coefficients))
    def present(self):
        if self.dimension() == 0:
            print(self.root+" is "+str(self.mapping[-1]))
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
                outpstr = str(self.mapping[0]) + '*x+' + str(self.mapping[1])
            else:
                inpstr = '('
                outpstr = ''
                for comp in range(self.dimension()):
                    inpstr += 'x'+str(comp)+','
                    outpstr += str(self.mapping[comp]) + '*x' + str(comp) + '+'
                inpstr = inpstr[:-1] + ')'
                if self.mapping[-1] != 0:
                    outpstr += str(self.mapping[-1])
                else:
                    outpstr = outpstr[:-1]
            retstr = "Function " + self.root + " maps " + domainstr + " by " + inpstr + ' -> ' + outpstr
            print(retstr)
    def reinforce(self,lexicon,supervisor):
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
            #print(word_cand.root)
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
                            #print('proposal ' + word_cand.root + ' ' + str(word_cand.mapping[-1]) + ' already covered by ' + word.root + ' ' + str(word.mapping[-1]))
                            proposal_is_new = False
                            break
                    if not proposal_is_new:
                        break
                if proposal_is_new and proposal.actual_dimension() == copy.actual_dimension():
                    #print('Can I also say '+ word_cand.root + '?')
                    for v in supervisor:
                        if word_cand.root == v.word:
                            #print('Supervisor: Yes')
                            cand_parse = advanced_parse(word_cand.mapping[-1],word_cand.root,lexicon,False,False)
                            if cand_parse.root == self.root:                                
                                if v.number != word_cand.mapping[-1]:
                                    print('LEARNING ERROR: ' + v.word + ' is ' + str(v.number) + ' but learner assumes ' + str(word_cand.mapping[-1]))
                                copy = proposal
                                #copy.present()
                            else:
                                pass
                                print('OK, but I think this is not related')
                            break
                    else:
                        pass
                        #print('Supervisor: No')
        return copy

def intlinsolve(base,image):
    #print('base: ',base)
    #print('image: ',image)
    try:
        return solve(base,image)[0]+[0]
    except IndexError:
        #print('constant needed')
        try:
            return solve([b+[1] for b in base],image)[0]
        except NotImplementedError:
            #print('unique solution')
            #return [round(i) for i in np.dot(np.linalg.pinv(np.array([b+[1] for b in base], dtype=np.float64),rcond=1e-15),image)]
            return [round(i) for i in np.dot(np.linalg.pinv(np.array([b+[1] for b in base], dtype=np.float64)),image)]
    except NotImplementedError:
        #print('unique solution')
        return [round(i) for i in np.dot(np.linalg.pinv(np.array(base, dtype=np.float64)),image)]+[0]

def cartesian_product(listlist):
    if len(listlist)==0:
        print('ERROR: empty input in product')
        return False
    for liste in listlist:
        if not isinstance(liste,list):
            liste=[liste]
    cp=[[x] for x in listlist[0]]
    for liste in listlist[1:]:
        cp=[a+[b] for a in cp for b in liste]
    return cp


def delatinized(string):
    #print(string)
    ad = AlphabetDetector()
    if not ad.is_latin(string):
        if not ad.is_cyrillic(string):
            if ad.is_cyrillic(string[0]) and not ad.is_cyrillic(string[-1]):
                #print('first part is cyrillic')
                for point in range(len(string)):
                    if not ad.is_cyrillic(string[:point]):
                        return string[:point-2]
            elif not ad.is_cyrillic(string[0]) and ad.is_cyrillic(string[-1]):
                #print('last part is cyrillic')
                for point in reversed(range(len(string))):
                    if not ad.is_cyrillic(string[point:]):
                        return string[point+2:]
            elif ad.is_latin(string[0]):
                #print('first part is latin')
                for point in range(len(string)+1):
                    if not ad.is_latin(string[:point]):
                        return string[point-1:] 
            elif ad.is_latin(string[-1]):
                #print('last part is latin')
                for point in reversed(range(len(string)+1)):
                    if not ad.is_latin(string[point:]):
                        return string[:point+1]
            else:
                return string
        else:
            return string
    else:
        return string
    
def create_lexicon(language):
    LEX=[]
    try:
        num2words(1, lang=language)
        for integer in list(range(1,1001))+[1002,1006,1100,1200,1206,7000,7002,7006,7100,7200,7206,10000,17000,17206,20000,27000,27006,27200,27206]:
            try:
                numeral=num2words(integer, lang=language)
                voc=Vocabulary(integer,numeral)
                LEX=LEX+[voc]
            except:
                pass
        return LEX
    except:
        try:
            #lanu=pd.read_csv(r'C:\Users\ikm\OneDrive\Desktop\NumeralParsingPerformance\Languages&NumbersData\Numeral.csv', encoding = "utf_16", sep = '\t')
            lanu=pd.read_csv(r'Numeral.csv', encoding = "utf_16", sep = '\t')
            df=lanu[lanu['Language']==language]
            biscriptual=False
            if ' ' in df.iloc[0,2]:
                biscriptual=True
            for i in range(len(df)):
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
                #print(numeral)
                numeral=numeral.replace('%',',')
                #print(numeral)
                voc=Vocabulary(i+1,numeral)
                LEX=LEX+[voc]
            return LEX
        except:
            raise NotImplementedError("Language "+language+" is not supported or spelled differently")

def proto_parse(number,numeral,lexicon,print_documentation,print_result): #parse a (int number,str numeral)-pair using (current) lexicon 'lexicon'. boolean print_documentation toggles documentation printout. boolean print_result toggles result printout
    #print('parse '+numeral)
    if print_documentation: print('parse '+numeral+' '+str(number))
    lex1=lexicon+[Vocabulary(number,numeral)] # so the new word itself is found at the end
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
                        #highlights=[highlight for highlight in highlights if not highlight.start>=start]# if a highlight is contained in new highlight, then remove it from list of highlights
                        for highlight in highlights[:]: #browse through present highlights
                            if highlight.start>=start: # if a highlight is contained in new highlight,...
                                if print_documentation: print("remove "+highlight.numeral)
                                highlights.remove(highlight) #then remove it from list of highlights
                        highlights=highlights+[Highlight(entry,start)] # add new highlight
                        if print_documentation: print('Unpacked: ['+','.join([str(highlight.numeral) for highlight in highlights])+']')
                    else:
                        if print_documentation: print(substring+' ≥' + str(entry.number) + '/2')
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
    #for highlight in highlights:
    #    decompstr=decompstr+str(highlight.number)+','
    decompstr=decompstr+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral)] for highlight in highlights],[0 for highlight in highlights]+[number])

def advanced_parse(number, word, lexicon, print_doc, print_result):
    if print_doc: print('parse '+word+' '+str(number))
    lexicon1 = []
    if len(lexicon) != 0 and isinstance(lexicon[0],Vocabulary):
        lexicon1 = lexicon
    else:
        for entry in lexicon:
            if isinstance(entry,Vocabulary):
                lexicon += [entry]
            else: 
                lexicon1 += entry.all_outputs_as_voc()
    lexicon1 = lexicon1+[Vocabulary(number,word)]    
    checkpoint = 0
    highlights=[]
    mult_found=False
    for end in range(0, len(word)+1):
        startrange=set(range(checkpoint, end))
        for highlight in highlights:
            startrange=startrange-set(highlight.hlrange())
            #print('remove '+str(range(highlight[3]+1,len(highlight[0]))))
        startrange=sorted(list(startrange))
        #print('startrange='+str(startrange))
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
                        if print_doc: print(substr+" is ≥" + str(number) + '/2')
                        mult_found=True
                        checkpoint=end
                        potential_highlight = None
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
                                        if print_doc: 
                                            print(word[laterstart:end]+" is <sqrt("+str(entry.number)+")") #print(word[laterstart:end]+" is FAC or SUM. If it would contain mult, its square would be larger than "+entry.word+'.')
                                            if potential_highlight:
                                                print("Ignore " + potential_highlight.word + " because " + word[laterstart:end] + " is its subnumeral.")
                                            subentry_found=True
                                        for highlight in reversed(highlights):
                                            if highlight.end() > laterstart:
                                                if print_doc: print("remove "+highlight[0])
                                                highlights.remove(highlight)
                                        highlights=highlights+[Highlight(Vocabulary(subentry.number,subentry.word),laterstart)]
                                        checkpoint = laterstart
                                        if print_doc: print('Unpacked: ',[highlight.numeral for highlight in highlights])
                                    else:
                                        if entry.number % subentry.number != 0 and 2*subentry.number < number:
                                            if print_doc: 
                                                print(word[laterstart:end]+" is at least <" + str(number) + "/2 and is no divisor of " + entry.word + ".") #print(word[laterstart:end]+" probably contains SUM. As "+subentry.word+' is no divisor of '+entry.word+', '+entry.word+' has to contain SUM. '+subentry.word+' cannot contain FAC*MULT, as it is smaller than half of '+entry.word+': And it cannot be FAC, as its square is larger than '+entry.word+'. So it is composed of SUM and MULT. If it turns out to be irreducible with the present properties, we assume it is SUM')
                                                if potential_highlight:
                                                    print("Ignore " + potential_highlight.word + " because " + word[laterstart:end] + " is its subnumeral.")
                                            potential_highlight = Highlight(Vocabulary(subentry.number,subentry.word),laterstart)
                                            potential_checkpoint = laterstart
                                        else:
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
            suspected_mult = sohi[0]
            if print_doc: print("remove "+suspected_mult.root+' because ' + sohi[0].root + ' + ' + sohi[1].root + " = " + word + " and "+ sohi[0].root + ' > ' + sohi[1].root + "so it is probably mult.")
            highlights.remove(suspected_mult)
        if highlights[0].number * highlights[1].number == number:
            sohi = sorted(highlights, key=lambda highlight: highlight.number)
            suspected_mult = sohi[0]
            if print_doc: print("remove "+suspected_mult.root+' because ' + sohi[0].root + ' * ' + sohi[1].root + " = " + word + " and "+ sohi[0].root + ' > ' + sohi[1].root + "so it is probably mult.")
            highlights.remove(suspected_mult)
    elif len(highlights) == 3:
        suspected_mult = max(highlights, key=lambda highlight: highlight.number)
        if suspected_mult.number**2 > number:
            #other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            other_numbers = [highlight for highlight in highlights if highlight != suspected_mult]
            if other_numbers[0].number * suspected_mult.number + other_numbers[1].number == number:
                if print_doc: print("remove "+suspected_mult.root+ " because " + other_numbers[0].root + " * " + suspected_mult.root + " + " + other_numbers[1].root + " = " + word + " and " + suspected_mult.root + " > " + other_numbers[0].root + " so it is probably mult.")
                highlights.remove(suspected_mult)
            if other_numbers[1].number * suspected_mult.number + other_numbers[0].number == number:
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
        
    #print(str(highlights)+wort+' '+str(zahl))
    root=word
    for highlight in reversed(highlights):
        root=root[0:highlight.start]+'_'+root[highlight.end():len(root)]
    decompstr = str(number)+'='+root+'('+','.join([str(highlight.number) for highlight in highlights])+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral)] for highlight in highlights],[0 for highlight in highlights]+[number])


def list_scfunctions(language):
    print('Parse numerals in '+language)
    lex=create_lexicon(language)
    set_of_scfunctions=[]
    irreducibles=[]
    for entry in lex:
        parse=advanced_parse(entry.number,entry.word,lex,False,True)
        if '_' in parse.root:
            function_known=False
            for pos in range(len(set_of_scfunctions)):
                if set_of_scfunctions[pos].root == parse.root:
                    set_of_scfunctions[pos]=set_of_scfunctions[pos].merge(parse)
                    #print(set_of_scfunctions[pos].root+' now has '+str(len(set_of_scfunctions[pos].inputrange))+' inputs.')
                    function_known=True
                    break
            if not function_known:
                set_of_scfunctions=set_of_scfunctions+[parse]
        else:
            irreducibles += [parse]
    print(language+' has '+str(len(set_of_scfunctions))+' number functions and '+str(len(irreducibles))+' irreducible numbers.')
    print('The number functions are:')
    printout=f""
    for scf in set_of_scfunctions:
        example=scf.sample()
        printout=printout+scf.root+':\t x -> '+str([round(coeff) for coeff in scf.mapping[:-1]])+'*x+'+str(round(scf.mapping[-1]))+', \t e.g. '+str(example.mapping[-1])+' is '+str(example.root)+'\n'
    print(printout)
    print('The irreducibles are: ' + ', '.join([scf.root+' ('+str(scf.mapping[-1])+')' for scf in irreducibles]))
    print('')
    return len(set_of_scfunctions) + len(irreducibles)
    
def old_list_scfunctions(language):
    #print('Trying old parser')
    lex=create_lexicon(language)
    set_of_scfunctions=[]
    irreducible_count=0
    for entry in lex:
        parse=proto_parse(entry.number,entry.word,lex,False,True)
        if '_' in parse.root:
            function_known=False
            for pos in range(len(set_of_scfunctions)):
                if set_of_scfunctions[pos].root == parse.root:
                    set_of_scfunctions[pos]=set_of_scfunctions[pos].merge(parse)
                    #print(set_of_scfunctions[pos].root+' now has '+str(len(set_of_scfunctions[pos].inputrange))+' inputs.')
                    function_known=True
                    break
            if not function_known:
                set_of_scfunctions=set_of_scfunctions+[parse]
        else:
            irreducible_count=irreducible_count+1
    print('The old parser had structured '+language+' into '+str(len(set_of_scfunctions))+' number functions and '+str(irreducible_count)+' irreducible numbers.')
    print('')
    print(language+' has '+str(len(set_of_scfunctions))+' number functions and '+str(irreducible_count)+' irreducible numbers.')
    print('The number functions are:')
    printout=f""
    for scf in set_of_scfunctions:
        example=scf.sample()
        printout=printout+scf.root+':\t x -> '+str([round(coeff) for coeff in scf.mapping[:-1]])+'*x+'+str(round(scf.mapping[-1]))+', \t e.g. '+str(example.mapping[-1])+' is '+str(example.mapping[-1])+'\n'
    print(printout)
    return len(set_of_scfunctions) + irreducible_count

def decompose_numeral(number,language,version='new'):
    lex = create_lexicon(language)
    for entr in list(reversed(lex))[-number:]:
        if entr.number == number:
            word = entr.word
            break
    if version == 'new':
        advanced_parse(number,word,create_lexicon(language),True,True)
    elif version == 'old':
        proto_parse(number,word,create_lexicon(language),True,True)
            
def decompose_numeral_custom(number,word,lexicon,version='new'):
    if isinstance(lexicon,str):
        try:
            lex = create_lexicon(lexicon)
        except:
            raise NotImplementedError("Assuming " + language + " is a language, it is not supported. Use a custom lexicon to test the decomposer on it")
    elif isinstance(lexicon,list):
        if len(lexicon)==0:
            lex = lexicon
        elif isinstance(lexicon[0],list):
            try:
                lex = [Vocabulary(l[0],l[1]) for l in lexicon]
            except:
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


import ast

def main():
    print("Welcome to Numeral Decomposer")
    #print("The decomposer is meant to decompose numerals, so that their decomposition can serve for Hurford's-Packing-Strategy like grammar inference.")
    #print("Arithmetic condition
    if True:
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
            raise ValueError("Unvalid choice")
            
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
                print("Try again by spelling the language exactly as in https://github.com/ikmMaierBTUCS/Numeral-Decomposer-1.1/blob/main/Languages.") 
                print("If your language has no predefined lexicon, you could customize a lexicon instead.")
                language = str(input("Type in the language of your numeral: "))
                lexicon = create_lexicon(language)
            if number > 1000:
                print("Warning: The predefined lexicon may not cover numbers over 1000. In order to assure a proper decomposition, add your numeral's subnumerals larger than 1000 to the lexicon.")
                print("E.g. when decomposing 'twenty-one thousand and two', add '[[21000,'twenty-one thousand'],[1000,'one thousand'],[1002,'one thousand and two']]'")
                extralexicon = ast.literal_eval(input("Type in the additional lexicon entries or type '[]' to skip: "))
                #print(extralexicon)
                lexicon += [Vocabulary(l[0],l[1]) for l in extralexicon]
        elif customize == 'y':
            print("Provide a lexicon for the decomposition of your numeral. It should include all subnumerals of your numeral. For each subnumeral that you do not include, you are making a decision that the decomposer is supposed to take.")
            print("Example: In order to decompose enoderdnuhowt (201) in backward English, you may want to provide [[1,'eno'],[100,'derdnuh'],[2,'owt'],[101,'enoderdnuh'],[200,'derdnuhowt']]")
            print("Example: In order to decompose neetxisderdnuhowt (216) in backward English, you may want to provide [[16,'neetxis'],[116,'neetxisderdnuh'],[6,'xis'],[106,'xisderdnuh'],[206,'xisderdnuhowt'],[100,'derdnuh'],[200,'derdnuhowt'],[2,'owt']].")
            lexicon = ast.literal_eval(input("Type in the lexicon of your numeral: "))
            lexicon = [Vocabulary(l[0],l[1]) for l in lexicon]
        else:
            print("Error: Unvalid choice")
        
        decompose_numeral_custom(number,numeral,lexicon,version)
        

if __name__ == "__main__":
    main()