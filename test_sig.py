import curses
def main(stdscr):
    p = curses.newpad(10, 10)
    try:
        p.noutrefresh(0, 0, 0, 0, 5, 5)
        res = "SUCCESS"
    except Exception as e:
        res = f"ERROR: {e}"
    with open("sig_res.txt", "w") as f:
        f.write(res)

if __name__ == '__main__':
    curses.wrapper(main)
