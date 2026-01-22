import json

notebook_path = "submission2.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1
content = content.replace("Computers['model_3'] = model_3", "computers['model_3'] = model_3")

# Fix 2
content = content.replace("rsq = model3.score(computers[['Units']],y)*100", "rsq = model3.score(computers[['Units']],Y)*100")

with open(notebook_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Notebook fixed.")
