import markdown
from markdown.extensions.toc import TocExtension
#references
## https://www.digitalocean.com/community/tutorials/how-to-use-python-markdown-to-convert-markdown-text-to-html
## https://python-markdown.github.io/extensions/toc/

def main(md_path):
    md_toc = markdown.Markdown(extensions=['toc'])
    md=markdown.Markdown()
    with open(md_path, 'r') as f:
        text = f.read()
        toc=md_toc.convert(text)
        html_body = md.convert(text)
    return md_toc.toc,html_body

if __name__ == '__main__':
    md_path='./templates/home.md'
    toc,html_body=main(md_path)
    with open('./templates/home.html', 'w+') as f:
        f.write('{% extends \'base.html\' %}\n')
        f.write('{% block toc %}\n')
        f.write(toc)
        f.write('{% endblock %}\n')
        f.write('{% block page_content %}\n')
        f.write(html_body)
        f.write('{% endblock %}')