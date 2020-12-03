from datasets import load_dataset


if __name__ == "__main__":
    dataset = load_dataset('glue', 'mrpc', split='train')
    print(len(dataset))

