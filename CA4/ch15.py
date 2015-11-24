import sys
INFINITY = sys.maxint

formatted_text = ''


def print_neatly(words, M):

 global formatted_text
 words_size = len(words)

 #Extra spaces table
 extra_spaces = [[0 for x in range(words_size+1)] for x in range(words_size+1)]

 # Cost per line
 cost_per_line = [[0 for x in range(words_size+1)] for x in range(words_size+1)]

 # Optimal arragement of words
 c = [0 for x in range(words_size+1)]

 # Reference of Start and End point of words per line
 p = [0 for x in range(words_size+1)]

 # Calculate extra spaces in single line
 i = 1
 while i <= words_size:
     extra_spaces[i][i] = M - len(words[i - 1])
     j = i + 1
     while j <= words_size:
         extra_spaces[i][j] = extra_spaces[i][j - 1] - len(words[j - 1]) - 1
         j += 1
     i += 1

 # Calculate per line cost
 i = 1
 while i <= words_size:
     j = i
     while j <= words_size:
         if extra_spaces[i][j] < 0:
             cost_per_line[i][j] = INFINITY
         elif j == words_size and extra_spaces[i][j] >= 0:
             cost_per_line[i][j] = 0
         else:
             cost_per_line[i][j] = extra_spaces[i][j]**3
         j += 1
     i += 1

 # Calculate minimum cost and their optimal arragement
 c[0] = 0
 j = 1
 while j <= words_size:
  c[j] = INFINITY
  i = 1
  while i <= j:
	  if c[i - 1] != INFINITY and cost_per_line[i][j] != INFINITY and (c[i - 1] + cost_per_line[i][j]) < c[j]:
		  c[j] = c[i - 1] + cost_per_line[i][j]
		  p[j] = i
	  i += 1
  j += 1

 # NEAT TEXT GENRATION

 generateNeatText(p, words_size,words)

 data = ''
 data += formatted_text
 formatted_text = ''
 data.rstrip('\n').strip()
 optimal_cost = c[words_size]

 return optimal_cost, data[:-1]

# Recursive Neat Text Generation
def generateNeatText(p,n,words):
    if p[n] == 1:
        k = 1;
    else:
        k = generateNeatText (p, p[n]-1,words) + 1

    global formatted_text
    formatted_text += formatIt(p[n],n,words)

    return k

# Generate Each line with words position
def formatIt(start, end, words):
    txt = ''
    i = start
    while i <= end:
        if i == end:
            txt += words[i - 1]
        else:
            txt += words[i - 1] + ' '
        i += 1
    txt += '\n'
    return txt