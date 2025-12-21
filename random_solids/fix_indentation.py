#!/usr/bin/env python3
"""
Fix indentation in Reconstruct_Solid.py
Add 4 spaces to lines 950-1592 to place them inside the for iteration loop
"""

def fix_indentation(input_file, output_file, start_line, end_line, spaces_to_add=4):
    """
    Add spaces to specific lines in a file
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        start_line: First line to indent (1-indexed)
        end_line: Last line to indent (1-indexed)
        spaces_to_add: Number of spaces to add
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    indent = ' ' * spaces_to_add
    
    # Process lines (convert to 0-indexed)
    for i in range(start_line - 1, min(end_line, len(lines))):
        # Only add indentation to non-empty lines
        if lines[i].strip():
            lines[i] = indent + lines[i]
    
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    print(f"Fixed indentation for lines {start_line}-{end_line}")
    print(f"Added {spaces_to_add} spaces to {end_line - start_line + 1} lines")
    print(f"Output written to: {output_file}")

if __name__ == '__main__':
    input_file = 'Reconstruct_Solid.py'
    output_file = 'Reconstruct_Solid.py.fixed'
    
    # Lines that need to be indented (inside the for loop)
    # From Step 1 (line 955) through the end of iteration closing logic (line 1592)
    # Lines 927-954 are already correctly indented (the for loop header and
    # the if/else that determines edges_to_process)
    start_line = 955
    end_line = 1592
    
    print("Fixing indentation in Reconstruct_Solid.py...")
    print(f"Processing lines {start_line} to {end_line}")
    
    fix_indentation(input_file, output_file, start_line, end_line, spaces_to_add=4)
    
    print("\nDone! Review the file and then:")
    print("  mv Reconstruct_Solid.py Reconstruct_Solid.py.backup")
    print("  mv Reconstruct_Solid.py.fixed Reconstruct_Solid.py")
