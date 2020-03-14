




def main():
    LOG_FILE="redis-mon.log"
    times = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            if "set" in line:
                times.append(float(line.lower().split()[0]))

    print(",".join(map(str, times)))


            
    return 0


if __name__ == "__main__":
    main()