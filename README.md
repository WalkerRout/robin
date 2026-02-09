# robin
Optimizing native compiler for the class of (partial) primitive recursive functions, using cranelift...
- Turing-complete by Church's thesis

### Usage
```
Usage: robin.exe [-o <output>] [--emit <emit>] [--opt-level <opt-level>] [--] <input>

robin - a compiler for the class of recursive functions

Positional Arguments:
  input             input source file (.rec)

Options:
  -o, --output      output file path (default: <stem>.o or <stem>.exe)
  --emit            emit type: obj or exe (default: exe)
  --opt-level       optimization: 0=none, 1=speed, 2=speed_and_size (default: 1)
  --help, help      display usage information
```

### Grammar
```
⟨program⟩   ::= ⟨decl⟩*

⟨decl⟩      ::= ⟨def⟩
             | ⟨eval⟩

⟨def⟩       ::= "def" ⟨ident⟩ "=" ⟨expr⟩ ";"

⟨eval⟩      ::= "eval" ⟨expr⟩ "(" ⟨args⟩ ")" ";"

⟨args⟩      ::= ε
             | ⟨nat⟩ ( "," ⟨nat⟩ )*

⟨expr⟩      ::= ⟨ident⟩ 
             | "const" "(" ⟨nat⟩ "," ⟨nat⟩ ")"
             | "s"
             | "id" "(" ⟨nat⟩ "," ⟨nat⟩ ")"
             | "Cn" "[" ⟨expr⟩ "," ⟨expr-list⟩ "]"
             | "Pr" "[" ⟨expr⟩ "," ⟨expr⟩ "]"
             | "Mn" "[" ⟨expr⟩ "]"

⟨expr-list⟩ ::= ⟨expr⟩ ( "," ⟨expr⟩ )*

⟨ident⟩     ::= ( letter | "_" ) ( letter | digit | "_" )*

⟨nat⟩       ::= digit+

⟨comment⟩   ::= "//" (any character except newline)* newline
```

### Common Functions
```
// Zero function: z(x) = 0
def z = const(1, 0);

// Predecessor: pred(0) = 0, pred(y') = y
def pred = Pr[const(0, 0), id(1,2)];

// Addition: add(x, 0) = x, add(x, y') = s(add(x, y))
def add = Pr[id(1,1), Cn[s, id(3,3)]];

// Multiplication: mult(x, 0) = 0, mult(x, y') = add(mult(x,y), x)
def mult = Pr[z, Cn[add, id(3,3), id(1,3)]];

// Exponentiation: exp(x, 0) = 1, exp(x, y') = mult(x, exp(x, y))
def exp = Pr[const(1, 1), Cn[mult, id(1,3), id(3,3)]];

// Factorial: fact(0) = 1, fact(y') = mult(y', fact(y))
def fact = Pr[const(0, 1), Cn[mult, Cn[s, id(1,2)], id(2,2)]];

// Truncated subtraction (monus): monus(x, 0) = x, monus(x, y') = pred(monus(x,y))
def monus = Pr[id(1,1), Cn[pred, id(3,3)]];

// Signum: sg(0) = 0, sg(y') = 1
def sg = Pr[const(0, 0), Cn[s, Cn[z, id(1,2)]]];

// Negated signum: sgbar(0) = 1, sgbar(y') = 0
def sgbar = Pr[const(0, 1), Cn[z, id(1,2)]];
```

### References
- Robinson, R. M. (1947). Primitive recursive functions. *Bulletin of the American Mathematical Society, 53*(10), 925–942.
- Gladstone, M. D. (1971). Simplifications of the recursion scheme. *The Journal of Symbolic Logic, 36*(4), 653–665. https://doi.org/10.2307/2272468
- Severin, D. E. (2008). Unary primitive recursive functions. *The Journal of Symbolic Logic, 73*(4), 1122–1138. https://doi.org/10.2178/jsl/1230396909

#### BibTeX Citations
```bibtex
@article{robinson-1947,
    author = {Raphael M. Robinson},
    title = {{Primitive recursive functions}},
    volume = {53},
    journal = {Bulletin of the American Mathematical Society},
    number = {10},
    publisher = {American Mathematical Society},
    pages = {925 -- 942},
    year = {1947},
}

@article{gladstone-1971,
    URL = {https://doi.org/10.2307/2272468},
    author = {M. D. Gladstone},
    journal = {The Journal of Symbolic Logic},
    number = {4},
    pages = {653--665},
    publisher = {Association for Symbolic Logic},
    title = {Simplifications of the Recursion Scheme},
    volume = {36},
    year = {1971}
}

@article{severin-2008,
     URL = {https://doi.org/10.2178/jsl/1230396909},
     author = {Daniel E. Severin},
     journal = {The Journal of Symbolic Logic},
     number = {4},
     pages = {1122--1138},
     publisher = {Association for Symbolic Logic},
     title = {Unary Primitive Recursive Functions},
     volume = {73},
     year = {2008}
}
```
