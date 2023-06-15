import os
import openai
from tqdm import tqdm
import csv
import argparse
from time import sleep

# openai.api_key = os.environ["OPEN_AI_KEY"]
openai.api_key = "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"

def sample_paraphrase(text):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f'''Paraphrase the paragraph below concisely and accurately:\n\n{text}'''}],
    temperature=0,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]

# print("My favorite film this year. Great characters and plot, and the direction and editing was smooth, visually beautiful, and interesting. <br /><br />Set in Barcelona, the film follows a year in the lives of six foreign graduate students and assorted others. Cultures and languages clash but hearts and lives intertwine. The leading role would never have been cast in Hollywood, but he carried the part perfectly. The characters were nicely developed and their interplay was honest and accurate. There were two especially noteworthy scenes, the climax was truly inspired. The film is sentimental, and the last ten minutes could have been cut, but it was wonderfully entertaining. <br /><br />I nearly didn't watch it, but did just to see Audrey Tautou. Her role although billed second or third was minor, and was outshined by several other characters. I wish more films like this were made. It brought to mind The Big Chill or The Breakfast Club. Don't start this movie late if you plan to go to bed 1/2 way through.")
# print("-"*100)
# print(sample_paraphrase("My favorite film this year. Great characters and plot, and the direction and editing was smooth, visually beautiful, and interesting. <br /><br />Set in Barcelona, the film follows a year in the lives of six foreign graduate students and assorted others. Cultures and languages clash but hearts and lives intertwine. The leading role would never have been cast in Hollywood, but he carried the part perfectly. The characters were nicely developed and their interplay was honest and accurate. There were two especially noteworthy scenes, the climax was truly inspired. The film is sentimental, and the last ten minutes could have been cut, but it was wonderfully entertaining. <br /><br />I nearly didn't watch it, but did just to see Audrey Tautou. Her role although billed second or third was minor, and was outshined by several other characters. I wish more films like this were made. It brought to mind The Big Chill or The Breakfast Club. Don't start this movie late if you plan to go to bed 1/2 way through."))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="train_dataset.csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        default="augmented.csv",
        type=str,
        required=True,
    )
    parser.add_argument("--start_from", required=False, default=0, type=int)
    parser.add_argument("--write_header", default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    with open(f'{args.input_file}', "r", newline="") as csv_read_file:
        permission = "a+" if os.path.exists(f'{args.output_file}') else "w+"
        with open(f'{args.output_file}', permission, newline="") as csv_write_file:
            csv_reader = csv.reader(csv_read_file)
            csv_writer = csv.writer(csv_write_file)
            count = 0

            for index, row in enumerate(tqdm(csv_reader)):
                text, label = row

                if index < args.start_from:
                    continue

                if index == 0 and args.write_header:
                    csv_writer.writerow(["original_text", "perturbed_text", "label"])
                
                elif count == 700:
                    print("Finished paraphrasing samples. Exiting")
                    break

                else:
                    try:
                        paraphrased = sample_paraphrase(text)
                        csv_writer.writerow([text, paraphrased, label])
                        count += 1
                    except Exception as e:
                        print("Sleeping...")
                        sleep(1)
                        continue