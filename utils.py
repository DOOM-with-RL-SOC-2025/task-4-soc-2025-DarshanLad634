# Converts position 1–16 to (x, y) coordinates on 4x4 grid
def pos_to_xy(pos):
    x = (pos - 1) % 4  # column (0-indexed)
    y = (pos - 1) // 4  # row (0-indexed)
    return x, y

# Converts (x, y) back to position in 1–16 range
def xy_to_pos(x, y):
    if 0 <= x < 4 and 0 <= y < 4:
        return y * 4 + x + 1
    return -1  # invalid/out-of-bounds

# Checks if a coordinate is inside the 4x4 grid
def in_bounds(x, y):
    return 0 <= x < 4 and 0 <= y < 4

# Finds intermediate positions lying between start and end positions (for pass)
def get_line_between(start, end):
    x1, y1 = pos_to_xy(start)
    x2, y2 = pos_to_xy(end)
    line = []

    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return []

    # Check each intermediate step between start and end
    for i in range(1, steps):
        x = round(x1 + dx * i / steps)
        y = round(y1 + dy * i / steps)
        if in_bounds(x, y):
            pos = xy_to_pos(x, y)
            if pos != start and pos != end:
                line.append(pos)
    return line

# Pretty-print the field for any given state (for debugging)
def pretty_print(state):
    if state == (-1, -1, -1, -1):
        print("[TERMINAL STATE]")
        return

    b1, b2, r, ball = state
    grid = [['.' for _ in range(4)] for _ in range(4)]

    # Helper: place player symbol on grid
    def place(pos, char):
        x, y = pos_to_xy(pos)
        grid[y][x] = char

    place(b1, 'B1' if ball == 1 else 'b1')  # Capital = with ball, small = without
    place(b2, 'B2' if ball == 2 else 'b2')
    place(r, 'R')

    # Print formatted grid
    for row in grid:
        print(' '.join(f'{cell:>2}' for cell in row))
