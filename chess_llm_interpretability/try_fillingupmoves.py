expanded_states = []
    move_index = 0
    idx = -1
    for i, char in enumerate(moves_expander):
        if char == ".":
            idx = i+1
            expanded_states.append(initial_states[min(move_index, len(initial_states) - 1)])
            break
        expanded_states.append(initial_states[min(move_index, len(initial_states) - 1)])
    
    toogle21 = True
    hacky_first = 0
    for char in moves_expander[idx:]:
        if toogle21:
            if char == " ":
                hacky_first+=1
                if hacky_first == 2:
                    move_index += 1
                    hacky_first = 0
                    toogle21 = False
        else:
            if char == ".":
                move_index += 1
                toogle21 = True
        expanded_states.append(initial_states[min(move_index, len(initial_states) - 1)])
    