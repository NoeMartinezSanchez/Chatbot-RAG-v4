import torch
import transformers
import psutil
import time

print('=== DIAGNÓSTICO DEL SISTEMA ===')
mem = psutil.virtual_memory()
print(f'RAM total: {mem.total / 1e9:.2f} GB')
print(f'RAM disponible: {mem.available / 1e9:.2f} GB')
print(f'RAM usada: {mem.percent}%')

print('\n=== VERSIONES ===')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')

if mem.available / 1e9 < 2:
    print('\n⚠️  ADVERTENCIA: Memoria disponible menor a 2GB')
    print('   La prueba será MUY lenta o podría fallar')
    print('   Considera:')
    print('   1. Cerrar más programas')
    print('   2. Usar use_quantization=False en CPU (no ayuda mucho)')
    print('   3. Aceptar que tomará varios minutos')

print('\n✅ Verificación completada')