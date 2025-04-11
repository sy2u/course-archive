def generate_readmemh_lst(filename, num_addresses=100, words_per_address=8):
    with open(filename, 'w') as file:
        for address in range(num_addresses):
            addr_str = f"@{address:02X}"  
            words = [f"{address * 16 + word:08X}" for word in range(1, words_per_address + 1)]
            file.write(f"{addr_str} {'_'.join(words)}\n")
            

if __name__ == "__main__":
    filename = "../testcode/memory.lst"
    generate_readmemh_lst(filename)
    print(f"Memory initialization written to {filename}")