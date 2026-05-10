import curses
def main(stdscr):
    p = curses.newpad(10, 10)
    with open("curses_inspect.txt", "w") as f:
        f.write(f"Type: {type(p)}\n")
        f.write(f"Has pnoutrefresh: {hasattr(p, 'pnoutrefresh')}\n")
        f.write(f"Has noutrefresh: {hasattr(p, 'noutrefresh')}\n")
        f.write(f"Methods: {[m for m in dir(p) if 'refresh' in m]}\n")

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except Exception as e:
        print(f"Error: {e}")
