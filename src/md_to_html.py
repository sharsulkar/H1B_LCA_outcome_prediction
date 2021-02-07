import markdown as md
from markdown.extensions.toc import TocExtension
#references
## https://www.digitalocean.com/community/tutorials/how-to-use-python-markdown-to-convert-markdown-text-to-html
## https://python-markdown.github.io/extensions/toc/

with open('./reports/notes/LSTM.md', 'r') as f:
    text = f.read()
    html = md.markdown(text,extensions=[TocExtension(baselevel=3)])

with open('./reports/notes/LSTM.html', 'w') as f:
    f.write(html)