import re
import numpy as np

def parse_faces_from_output(output_lines, section_header):
    """
    Parse faces and their vertex sequences from a section of the output.
    Returns a list of (vertex_sequence, face_index) tuples.
    """
    faces = []
    in_section = False
    current_vertices = []
    for line in output_lines:
        if section_header in line:
            in_section = True
            continue
        if in_section:
            if re.match(r"\s*Face \d+: ", line):
                if current_vertices:
                    faces.append(current_vertices)
                    current_vertices = []
            match = re.findall(r"\(([-\d\., ]+)\)", line)
            if match:
                for group in match:
                    coords = [float(x) for x in group.split(',')]
                    current_vertices.append(tuple(np.round(coords, 3)))
            if 'Successfully extracted' in line or '3D SOLID VISUALIZATION' in line:
                if current_vertices:
                    faces.append(current_vertices)
                break
    return faces

    def parse_faces_from_output(output_lines, section_header):
        """
        Parse faces and their vertex sequences from a section of the output.
        Returns a dict mapping face number to vertex sequence.
        """
        faces = {}
        in_section = False
        current_vertices = []
        current_face = None
        for line in output_lines:
            if section_header in line:
                in_section = True
                continue
            if in_section:
                m = re.match(r"\s*Face (\d+):", line)
                if m:
                    if current_face is not None and current_vertices:
                        faces[current_face] = current_vertices
                    current_face = int(m.group(1))
                    current_vertices = []
                match = re.findall(r"\(([-\d\., ]+)\)", line)
                if match:
                    for group in match:
                        coords = [float(x) for x in group.split(',')]
                        current_vertices.append(tuple(np.round(coords, 3)))
                if 'Successfully extracted' in line or '3D SOLID VISUALIZATION' in line:
                    if current_face is not None and current_vertices:
                        faces[current_face] = current_vertices
                    break
        return faces

def parse_classified_faces(output_lines):
    """
    Parse classified faces from the ENHANCED FACE CLASSIFICATION section.
    Returns a dict mapping vertex sequence (as tuple) to classification.
    """
    faces = {}
    in_section = False
    current_vertices = []
    current_class = None
    for line in output_lines:
        if 'ENHANCED FACE CLASSIFICATION' in line:
            in_section = True
            continue
        if in_section:
            if line.strip().startswith('Face F'):
                if current_vertices and current_class:
                    faces[tuple(current_vertices)] = current_class
                    current_vertices = []
                current_class = 'visible' if 'dot_product=0.577' in line or 'dot_product=1.000' in line else 'hidden'
            match = re.findall(r"\(([-\d\., ]+)\)", line)
            if match:
                for group in match:
                    coords = [float(x) for x in group.split(',')]
                    current_vertices.append(tuple(np.round(coords, 3)))
            if 'area:' in line and current_vertices and current_class:
                faces[tuple(current_vertices)] = current_class
                current_vertices = []
            if 'Step 1:' in line or 'Unit projection normal:' in line:
                continue
            if 'Step' in line and 'Initial' not in line:
                break
    return faces

    def parse_classified_faces(output_lines):
        """
        Parse classified faces from the ENHANCED FACE CLASSIFICATION section.
        Returns a dict mapping face number to classification.
        """
        faces = {}
        in_section = False
        current_face = None
        current_class = None
        for line in output_lines:
            if 'ENHANCED FACE CLASSIFICATION' in line:
                in_section = True
                continue
            if in_section:
                m = re.match(r"Face F(\d+): dot_product=([\-\d\.]+)", line.strip())
                if m:
                    current_face = int(m.group(1))
                    dot = float(m.group(2))
                    # Use dot product sign to determine class
                    if dot > 0:
                        current_class = 'visible'
                    else:
                        current_class = 'hidden'
                    faces[current_face] = current_class
                if 'Step' in line and 'Initial' not in line:
                    break
        return faces

def compare_vertex_sequences(faces1, faces2):
    """
    Compare two lists of vertex sequences. Returns a list of mismatches.
    """
    mismatches = []
    for seq1 in faces1:
        found = False
        for seq2 in faces2:
            if len(seq1) == len(seq2) and all(np.allclose(a, b, atol=0.01) for a, b in zip(seq1, seq2)):
                found = True
                break
        if not found:
            mismatches.append(seq1)
    return mismatches

def main():
    with open('output.txt', 'r') as f:
        lines = f.readlines()

    # Parse faces from the Matplotlib/3D visualization section
    faces_plot = parse_faces_from_output(lines, '3D SOLID VISUALIZATION')
    # Parse faces from the classification section
    faces_class = parse_classified_faces(lines)

    # Compare by matching vertex sequences
    mismatches = []
    for plot_seq in faces_plot:
        found = False
        for class_seq in faces_class.keys():
            if len(plot_seq) == len(class_seq) and all(np.allclose(a, b, atol=0.01) for a, b in zip(plot_seq, class_seq)):
                found = True
                break
        if not found:
            mismatches.append(plot_seq)

    print(f"Total faces in plot section: {len(faces_plot)}")
    print(f"Total classified faces: {len(faces_class)}")
    print(f"Faces in plot section not found in classification: {len(mismatches)}")
    for i, seq in enumerate(mismatches, 1):
        print(f"Mismatch {i}: {seq}")

    # Now check for classification mismatches
    for class_seq, classification in faces_class.items():
        found = False
        for plot_seq in faces_plot:
            if len(plot_seq) == len(class_seq) and all(np.allclose(a, b, atol=0.01) for a, b in zip(plot_seq, class_seq)):
                found = True
                break
        if not found:
            print(f"Classified face not found in plot section: {class_seq}")

        with open('output.txt', 'r') as f:
            lines = f.readlines()

        # Parse faces from the Matplotlib/3D visualization section
        faces_plot = parse_faces_from_output(lines, '3D SOLID VISUALIZATION')
        # Parse faces from the classification section
        faces_class = parse_classified_faces(lines)

        print(f"Total faces in plot section: {len(faces_plot)}")
        print(f"Total classified faces: {len(faces_class)}")

        # Compare by face number
        all_face_nums = sorted(set(faces_plot.keys()) | set(faces_class.keys()))
        for face_num in all_face_nums:
            plot_seq = faces_plot.get(face_num)
            classif = faces_class.get(face_num)
            if plot_seq is None:
                print(f"Face {face_num} missing in plot section.")
            elif classif is None:
                print(f"Face {face_num} missing in classification section.")
            else:
                # Optionally, print both for manual inspection
                print(f"Face {face_num}: classification = {classif}, vertices = {plot_seq}")

if __name__ == "__main__":
    main()
