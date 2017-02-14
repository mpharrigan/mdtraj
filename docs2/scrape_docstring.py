import ast


class ClassDocstringScraper(ast.NodeVisitor):
    def __init__(self, classname):
        self.classname = classname
        self.class_docstring = ""
        self.function_docstrings = []

    def visit_FunctionDef(self, node):
        pass

    def visit_ClassDef(self, node):
        if node.name != self.classname:
            return

        self.class_docstring = ast.get_docstring(node)

        for n2 in ast.iter_child_nodes(node):
            if isinstance(n2, ast.FunctionDef):
                if n2.name.startswith('_'):
                    continue
                doc = ast.get_docstring(n2)
                if doc is None:
                    doc = ""
                self.function_docstrings += [(n2.name, n2.args, doc)]

import re
def fix_definition_list_multiline(s):
    return re.sub(r'(\w+) : (.*)\n    (.*)$(\n    (.*)$)*', r'\1\n:   \2\n:   \3\4\n', s, flags=re.MULTILINE)
def fix_definition_list_oneline(s):
    return re.sub(r'(\w+) : (.*)', r'\1\n:   \2\n', s)

def fix(s):
    s = fix_definition_list_multiline(s)
    s = fix_definition_list_oneline(s)
    return s

from subprocess import run, PIPE
class ClassDoc:
    def __init__(self, fn, classname):
        self.fn = fn
        self.classname = classname

        scraper = ClassDocstringScraper(classname)
        with open(fn) as file:
            root = ast.parse(file.read(), filename=fn)
        scraper.visit(root)
        self.scraper = scraper

    def write_markdown(self, fn):
        with open(fn, 'w') as file:
            file.write("{}\n".format(self.classname))
            file.write("=" * len(self.classname) + '\n')
            doc = fix(self.scraper.class_docstring)
            file.write(doc)
            file.write('\n\n')

            for name, args, doc in self.scraper.function_docstrings:
                top = "`{}({})`".format(name, ', '.join(arg.arg for arg in args.args))
                divy = '=' * len(top)
                doc = fix(doc)
                file.write("{}\n{}\n{}\n\n\n".format(top, divy, doc))

    def write_html(self, fn, markdown_fn):
        proc = run(['C:/Ruby22-x64/bin/kramdown.bat', markdown_fn], stdout=PIPE, stderr=PIPE, universal_newlines=True)
        html = proc.stdout
        print(proc.stderr)

        template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>MDTraj</title>
            <style>
            dt {{
              font-weight: bold;
            }}

            dt+dd {{
              color: blue;
            }}

            dt+dd+dd {{
              color: red;
            }}

            </style>
        </head>
        <body>
        <div id="content" style="margin-left: 10rem; margin-right:10rem;">
        {}
        </div>
        </body>
        </html>
        """

        with open(fn, 'w') as file:
            file.write(template.format(html))

    def write(self):
        md_path = "docs2/{}.md".format(self.classname)
        html_path = "docs2/{}.html".format(self.classname)
        self.write_markdown(md_path)
        self.write_html(html_path, md_path)

ClassDoc('mdtraj/core/trajectory.py', 'Trajectory').write()
ClassDoc('mdtraj/core/coordinates.py', 'Coordinates').write()
ClassDoc('mdtraj/formats/netcdf.py', 'NetCDFCoordinates').write()
