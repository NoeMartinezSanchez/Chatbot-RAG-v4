# generate_tree.py
import os
from pathlib import Path

def should_exclude(path_name):
    """Define qué carpetas y archivos excluir"""
    exclude_patterns = {
        'venv', '__pycache__', '.git', 'node_modules', 
        '.vscode', '.idea', 'dist', 'build', '.pytest_cache',
        '*.pyc', '*.pyo', '*.egg-info'
    }
    
    for pattern in exclude_patterns:
        if pattern in str(path_name):
            return True
    return False

def generate_tree(directory, prefix='', max_depth=4, current_depth=0):
    """Genera un árbol de directorios limpio"""
    if current_depth > max_depth:
        return []
    
    lines = []
    items = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    # Filtrar items excluidos
    items = [item for item in items if not should_exclude(item.name)]
    
    for index, item in enumerate(items):
        is_last = index == len(items) - 1
        connector = '└── ' if is_last else '├── '
        
        if item.is_dir():
            lines.append(f"{prefix}{connector}📁 {item.name}/")
            extension = '    ' if is_last else '│   '
            lines.extend(generate_tree(item, prefix + extension, max_depth, current_depth + 1))
        else:
            lines.append(f"{prefix}{connector}📄 {item.name}")
    
    return lines

if __name__ == "__main__":
    project_name = Path.cwd().name
    print(f"📦 {project_name}/")
    tree_lines = generate_tree('.', max_depth=2)
    
    output = f"📦 {project_name}/\n" + '\n'.join(tree_lines)
    print(output)
    
    # Guardar en archivo
    with open('estructura_proyecto.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    
    print("\n✅ Estructura guardada en 'estructura_proyecto.txt'")