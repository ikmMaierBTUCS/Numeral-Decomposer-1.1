# Numeral-Decomposer-1.1
Improved version of Numeral Decomposer 1.0, see https://github.com/ikmMaierBTUCS/Numeral-decomposer-1.0

It can decompose numeral words into a function with inputs, e.g. 

'two hundred and twenty-seven' --> '_ hundred and \_'(2,27). 

Given that all numerals of the shape 'X hundred and Y' are decomposed to '_ hundred and \_'(X,Y), they can be grouped to one single function 

'_ hundred and \_': {1,...,9}x{1,...,99}, (x,y)->100x+y.

The decomposition algorithm is based on arithmetic criteria that do not only work for base 10 numeral systems, but almost any system, see [results/performance_analysis.txt](results/performance_analysis.txt).

## Try it out
```
pip install -r requirements.txt
python numdec.py
```

## Repository contents
- `numdec.py` contains the Numeral Decomposer 1.1 (function `advanced_parse`) and its prototype 1.0 (function `proto_parse`).
  Function `decompose_numeral` is easy to use. Just pass an int number and a language* to see a fully documented decomposition of the numeral.
- `data/` contains the numerals of languages that are not covered by [num2words](https://pypi.org/project/num2words/).
  `Numeral.csv` contains the numerals from [Languages and Numbers](https://www.languagesandnumbers.com).
- `results/performance_analysis.txt` is the output of the script `performance_analysis.py`. For 260 languages, it summarizes what lexicon of functions is produced by the Numeral Decomposer 1.1 and 1.0.

In the plots, we present the lexicon sizes of the numeral decomposers, i.e. the numbers of functions that the numeral decomposers structure a language's numeral system into.
In `results/lexicon_sizes.png`, the blue line shows the lexicon sizes of the Numeral Decomposer 1.1 in ascending order from left to right. The green line shows the lexicon sizes of the Numeral Decomposer 1.0 in ascending order from left to right. For the sake of better presentation, we removed the 2 largest out of 277 values from the plot for both versions.

In `results/lexicon_size_ratios.png`, we plotted by which ratios the Numeral Decomposer 1.1 reduced lexica in comparison to the Numeral Decomposer 1.0.

*Pass a language by writing one of the strings listed in [languages.txt](languages.txt).
