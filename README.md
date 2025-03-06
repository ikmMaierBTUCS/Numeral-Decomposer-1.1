# Numeral-Decomposer-1.1
Improved version of Numeral Decompose 1.0, see https://github.com/ikmMaierBTUCS/Numeral-decomposer-1.0

It can decompose numeral words into a function with inputs, e.g. 

'two hundred and twenty-seven' --> '_ hundred and \_'(2,27). 

Given that all numerals of the shape 'X hundred and Y' are decomposed to '_ hundred and \_'(X,Y), they can be grouped to one single function 

'_ hundred and \_': {1,...,9}x{1,...,99}, (x,y)->100x+y.

The decomposition algorithm is based on arithmetic criteria that do not only work for base 10 numeral systems, but almost any system, see 'Performance Analysis'.

Try it out by
1. making sure that you have installed the packages ```numpy, num2words, sympy, diophantine, alphabet_detector, pandas``` ,
2. downloading the folder numdec_app,
3. navigating into it and
4. prompting ```python numdec.py```.

'Code.py' is a Python script containing the function advanced_parse, which represents the Numeral Decomposer 1.1.
Function decompose_numeral is easy to use. Just pass an int number and a language* to see a fully documented decomposition of the numeral.

'Performance Analysis' is the output of the script 'Code for Performance Analysis'. For 260 languages, it summarizes what lexicon of functions is produced by the Numeral Decomposer 1.1.

In the plots, we present the lexicon sizes of the numeral decomposers, i.e. the numbers of functions that the numeral decomposers structure a language's numeral system into.
In 'Lexicon sizes of Versions 1.0 and 1.1 plotted.png', the blue line shows the lexicon sizes of the Numeral Decomposer 1.1 in arising order from left to right. The green line shows the lexicon sizes of the Numeral Decomposer 1.0 in arising order from left to right. For the sake of better presentation, we removed the 2 largest out of 277 values from the the plot each for both versions.

In the plot 'Ratios of Version 1.1 lexicon sizes to Version 1.0 lexicon sizes.png', we plotted by which ratios the Numeral Decomposer 1.1 reduced lexica in comparison to the Numeral Decomposer 1.0.

*Pass a language by writing one the strings listed in 'Languages'. For some of the languages you also need to download 'Numeral.csv' and link its path to FILE_PATH_OF_LANGUAGESANDNUMBERS_DATA_CSV in the code.
