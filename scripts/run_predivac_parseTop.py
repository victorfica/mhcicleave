import subprocess
import os
import pandas as pd
from pathlib import Path

def run_predivac(input_file: str, mode: str, predivac_path: str = 'predivac', allele: str = None,
                  population: str = None, mhc: str = None, pct: int = None, ppr: int = None,
                 verbosity: int = 1, outdir = "results/predivac") -> str | None:
    """
    Un wrapper de Python para ejecutar la herramienta Predivac-3.0.

    Esta función construye el comando basado en los parámetros proporcionados,
    lo ejecuta y devuelve la ruta del archivo de salida esperado si tiene éxito.

    Args:
        input_file (str): La ruta al archivo de entrada (fasta o lista de péptidos).
        mode (str): El modo de ejecución ('peptide', 'protein', 
                    'epitope_discovery', 'epitope_optimization').
        predivac_path (str, optional): La ruta al ejecutable de predivac. 
                                       Por defecto es 'predivac'.
        allele (str, optional): El alelo MHC para los modos 'peptide' y 'protein'. 
                                E.g., 'DRB1*01:01'.
        population (str, optional): La población para los modos 'epitope_discovery' 
                                    y 'epitope_optimization'. E.g., 'Chile'.
        mhc (str, optional): La clase o locus MHC. E.g., 'I', 'II', 'A,B,DRB1'.
        pct (int, optional): El umbral de cobertura poblacional (PCT) para
                             'epitope_discovery'.
        ppr (int, optional): El umbral de Peptide Percentile Rank (1-5).
        verbosity (int, optional): Nivel de verbosidad (0, 1, o 2 para -v, -vv).

    Returns:
        str | None: La ruta al archivo de salida si la ejecución fue exitosa,
                    de lo contrario, None.
    """

    if not os.path.exists(input_file):
        print(f"Error: El archivo de entrada no se encuentra en '{input_file}'")
        return None

    command = [predivac_path, input_file, mode]

    if mode in ['peptide', 'protein'] and allele:
        command.extend(['-a', allele])
    elif mode in ['epitope_discovery', 'epitope_optimization']:
        if population:
            command.extend(['--population', population])
        if mhc:
            command.extend(['--mhc', mhc])
        if mode == 'epitope_discovery' and pct is not None:
            command.extend(['--pct', str(pct)])

    if ppr is not None:
        command.extend(['-ppr', str(ppr)])
    if verbosity > 0:
        command.append('-' + 'v' * verbosity)
        
    print(f"Ejecutando comando: {' '.join(command)}")

    # 4. Determinar el nombre del archivo de salida esperado
    input_base_name = Path(input_file).stem
    output_filename = None

    if mode == 'epitope_discovery' and population and mhc and pct is not None:
        # Formato: P1_gp_epi_discovery_Chile_II_30.out
        output_filename = f"P1_{input_base_name}_epi_discovery_{population}_{mhc}_{pct}.out"
        
    elif mode == 'protein' and allele:
        # Basado en tu feedback, el formato es: 'P1_gp_prot_HLA-DRB1_0101.out'
        # para un alelo como 'DRB1*01:01'.
        # La nueva lógica añade 'HLA-', cambia '*' por '_' y elimina ':'.
        formatted_allele = "HLA-" + allele.replace('*', '_').replace(':', '')
        output_filename = f"P1_{input_base_name}_prot_{formatted_allele}.out"
        
    # else:
    #     print(f"Advertencia: No se pudo determinar el patrón del nombre de archivo de salida para el modo '{mode}' con los parámetros dados.")
    #     # Podrías añadir aquí la lógica para otros modos como 'peptide' o 'epitope_optimization'
    #     # si descubres sus patrones de nombrado.
    #     return None # Terminamos si no podemos predecir el nombre del archivo

    # 5. Ejecutar el comando (sin cambios)
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        print("¡Predivac se ejecutó exitosamente!")
        print("Salida estándar (stdout):")
        print(process.stdout)

        return output_filename
        # if os.path.exists(output_filename):
        #     return output_filename
        # else:
        #     print(f"Error: La ejecución pareció exitosa, pero no se encontró el archivo de salida esperado '{output_filename}'.")
        #     print("Por favor, verifica el nombre de archivo que genera predivac.")
        #     return None

    except FileNotFoundError:
        print(f"Error: El comando '{predivac_path}' no fue encontrado.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar Predivac (código de retorno {e.returncode}):")
        print(f"Stderr: {e.stderr}")
        return None

def parse_top(predivac_out, top_n = 1):
    
    predivac_epitope_disc = pd.read_csv(predivac_out)
    predivac_epitope_disc = predivac_epitope_disc.sort_values(by='coverage',ascending=False)
    print(predivac_epitope_disc)
    top_epitopes = predivac_epitope_disc.head(1).iloc[:,0:2].values.tolist()
    
    if not top_epitopes or len(top_epitopes) < top_n:
        raise ValueError(f"Could not select {top_n} epitopes. Found only {len(top_epitopes)}.")
    
    return top_epitopes

input_file = './Q76R36.fasta'       
mode = 'epitope_discovery'
population = 'Chile'
mhc = ['I','II']
threshold = 30
verbosity = 1

predivac_results = []
for m in mhc:
    predivac_out = run_predivac(input_file=input_file, mode=mode, population=population, mhc=m, pct=threshold, verbosity=verbosity)
    top_epitopes = parse_top(predivac_out, top_n=1)
    # ids
    protid = predivac_out.split("_")[1] # the fasta id comes after P1_

    id_epitopes = [(f"{protid}_E{i}_{round(p[1])}",p[0],m) for i,p in enumerate(top_epitopes,1)]
    [predivac_results.append(e) for e in id_epitopes]
    #predivac_results.append([protid,top_epitopes,m])
    
top_df = pd.DataFrame(predivac_results, columns=["id", "peptide", "mhc"]).explode("peptide")
top_df.to_csv(f"{protid}_discovery_top1.csv", index = None)