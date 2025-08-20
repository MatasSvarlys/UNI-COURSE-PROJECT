def collide_vec_to_rect(start_pos, movment_vector, rectangle):
    # Example using clipline().
    clipped_line = rectangle.clipline(movment_vector)

    if clipped_line:
        # If clipped_line is not an empty tuple then the line
        # collides/overlaps with the rect. The returned value contains
        # the endpoints of the clipped line.
        start, _ = clipped_line
        x1, y1 = start
        
        dx = movment_vector[2] - movment_vector[0]
        dy = movment_vector[3] - movment_vector[1]
        if dx != 0:
            t = (x1 - start_pos[0]) / dx
        elif dy != 0:
            t = (y1 - start_pos[1]) / dy
        else:
            t = 0  # No movement

