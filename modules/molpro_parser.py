

def parse_atoms(molpro_out):
    """ 
    Parses the atoms with coordinates and symbol from a molpro output file
    """
    with open(molpro_out, 'r') as f:
        lines = f.readlines()
        # Search for Atomic Coordinates in line
        switch = False

        atomic_coordinates = []

        for i, line in enumerate(lines):
            if "Atomic Coordinates" in line:
                switch = True
                continue
            if switch:
                if "Gradient norm" in line:
                    break
                atomic_coordinates.append(line.strip())

        # Remove first entry the header
        atomic_coordinates = list(filter(None, atomic_coordinates))
        # Remove header
        atomic_coordinates = atomic_coordinates[1:]
        
        molecules = {} # Dictionary to hold molecule data
        for atom in atomic_coordinates:
            parts = atom.split()
            # First part is numbering, second part is symbol, fourth fifth and sixth are coordinates
            number = int(parts[0])
            symbol = parts[1]
            atomic_symbol = str(symbol) + str(number)
            x = float(parts[3])
            y = float(parts[4])
            z = float(parts[5])
            coordinates = (x, y, z)
            molecules[atomic_symbol] = coordinates
    return molecules

def parse_normal_modes(molpro_out):
    """ 
    Function to parse the normal modes from a molpro output file. The Normal Modes are given in blocks of 5
    """
    with open(molpro_out, "r") as f:
        lines = f.readlines()

        # Find all normal mode sections
        mode_blocks = []
        current_block = []
        in_block = False
        for line in lines:
            if "Normal Modes" in line and "low/zero frequencies" not in line:
                in_block = True
                current_block = [line]
                continue
            elif "Normal Modes of low/zero frequencies" in line and in_block:
                in_block = False
                if current_block:
                    mode_blocks.append(current_block)
                continue
            elif in_block:
                current_block.append(line)

        # Combine all blocks into a continious block
        full_block = []
        for block in mode_blocks:
            full_block.extend(block)
        
        normal_modes = {}

        # First find all mode headers (may be in multiple lines)
        mode_headers = []
        mode_lines = []
        

        for line in full_block:
            if line.startswith(" " * 12) and not any(x in line for x in ["Wavenumbers", "Intensities"]):

               # Clean up the line by replacing multiple spaces
               cleaned_line = " ".join(line.strip().split())
               mode_lines.append(cleaned_line)

        combined_mode_line = " ".join(mode_lines)
        
        # Parse mode numbers and symmetry
        mode_info = []
        parts = combined_mode_line.split()
        i = 0
        while i < len(parts):
            if parts[i].isdigit():
                mode_num = int(parts[i])
                symmetry = parts[i+1] if i+1 < len(parts) else ""
                mode_info.append((mode_num,symmetry))
                i +=2
            else:
                i += 1
        num_modes = len(mode_info)

        # parse wavenumbers and intensities
        wavenumbers = []
        intensities_km = []
        intensities_rel = []
        for line in full_block:
            if "Wavenumbers" in line:
                parts = line.split()
                new_wavenumbers = list(map(float, parts[2:2+num_modes]))
                wavenumbers.extend(new_wavenumbers)
            elif "Intensities [km/mol]" in line:
                parts = line.split()
                new_intensities = list(map(float,parts[2:2+num_modes]))
                intensities_km.extend(new_intensities)
            elif "Intensities [relative]" in line:
                parts = line.split()
                new_intensities = list(map(float,parts[2:2+num_modes]))
                intensities_rel.extend(new_intensities)

        # Some error printing
        if len(wavenumbers) != num_modes:
            raise ValueError(f"Expected {num_modes} wavenumbers, got {len(wavenumbers)}")
        if len(intensities_km) != num_modes:
            raise ValueError(f"Expected {num_modes} km/mol intensities, got {len(intensities_km)}")
        if len(intensities_rel) != num_modes:
            raise ValueError(f"Expected {num_modes} [relative] intensities, got {len(intensities_rel)}")
        
        for i, (mode_num, sym) in enumerate(mode_info):
            normal_modes[mode_num] = {
                "symmetry": sym,
                "wavenumber": wavenumbers[i] if i < len(wavenumbers) else 0.0,
                "intensity_km_mol": intensities_km[i] if i < len(intensities_km) else 0.0,
                "intensity_relative": intensities_rel[i] if i < len(intensities_rel) else 0.0,
                "displacements": {}
            }
        
        current_block_modes = [] # Track modes in current block
        current_block_size = 5 # Molpro Block size

        for line in full_block:
            if not line.strip() or any(x in line for x in ["Normal Modes", "Wavenumbers", "Intensities"]):
                continue
            
            # Check if its a mode header line
            if line.startswith(" " * 12) and not any(x in line for x in ["Wavenumbers", "Intensities"]):
                # This is new block of modes
                cleaned_line = " ".join(line.strip().split())
                parts = cleaned_line.split()
                current_block_modes = []
                i = 0
                while i < len(parts):
                    if parts[i].isdigit:
                        mode_num = int(parts[i])
                        current_block_modes.append(mode_num)
                        i += 2 # Skip symmetry label
                    else:
                        i +=1
                continue

            # Now process the displacement lines
            parts = line.split()
            if len(parts) < 2: # Skip lines without data
                continue

            label = parts[0]
            values = list(map(float,parts[1:1 + len(current_block_modes)])) # only take values of current block
            

            # Parse the atom info
            element = "".join([c for c in label if c.isalpha() and c not in ["X","Y","Z"]])
            atom_num = "".join([c for c in label if c.isdigit()])
            direction = label[len(element):-len(atom_num)].lower()
            atom_name = f"{element}{atom_num}" if atom_num else element

            
            # Add displacment for modes in current block
            for i, mode_num in enumerate(current_block_modes):
                if mode_num not in normal_modes:
                    continue # Skip if mode wasn't properly registered
                if atom_name not in normal_modes[mode_num]["displacements"]:
                    normal_modes[mode_num]["displacements"][atom_name] = {}
                normal_modes[mode_num]["displacements"][atom_name][direction] = values[i]
        return normal_modes