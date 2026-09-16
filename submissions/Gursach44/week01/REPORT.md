# Week 1 report

## Setup
- OS: Windows 11
- Installation method: fallback (pip)
- Coding agent: Claude

## Agent interaction
- Main task-level requests: 
    - ran setup (took a while, so manually created individual directories etc. at this point)
    - ran preflight
    - inspected data schemas
    - implemented exercise 1 in submission folder and asked to explain failures before modifying code
    - similar for exercise 3
- One useful suggestion: checker only worked for exercise 1+3 out of the box. Claude decided to copy the checker logic into inline python to manually check exercise 1 only, before completing exercise 3, which seemed to work well.
- One error, weak assumption, or unnecessary change: Nothing major, Claude was just a bit self-conscious about the size of the legend but it was not too large so I manually made it larger.

## Results and verification
- Summary result: 
    - exercise 1: everything matches ground truth file
    - exercise 3: object plotted successfully. 
- Object plotted: 252646, shows 2 intervals of no notable activity (baseline measurements), then a clear spike in luminosity in all passbands
- Commands run: essentially asking Claude to please run the checker
- Checker result: passed
- I inspected `lightcurve.png`: yes
- One check I performed myself: that the correct object was plotted and it was in fact plotting the different passbands

## Reflection
- One statement supported directly by the data: there is a clear flare (supernova) peaking a bit before MJD 60600
- One statement requiring further analysis: which passbands are brighter and whether the order is constant in time

