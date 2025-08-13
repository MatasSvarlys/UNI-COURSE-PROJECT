import argparse

def generate_map(mode):
    with open("map.txt", "w") as f:
        for i in range(15):
            if mode == "box":
                row = []
                for j in range(20):
                    if i == 0 or i == 14 or j == 0 or j == 19:
                        row.append("1")
                    else:
                        row.append("0")
                f.write(" ".join(row) + "\n")
            elif mode == "lastrow":
                if i == 14:
                    row = " ".join(["1"] * 20)
                else:
                    row = " ".join(["0"] * 20)
                f.write(row + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["box", "lastrow"],
        required=True,
        help="Choose 'perimeter' for a box of 1s or 'lastrow' for 1s on the last row"
    )
    args = parser.parse_args()
    generate_map(args.mode)