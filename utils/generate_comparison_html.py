"""
Function to generate an HTML file visualizing comparison results from multiple methods

Usage:
    python utils/generate_comparison_html.py  [--input_file INPUT_FILE] [--output_html OUTPUT_HTML]

Arguments:
    - input_file: The file that saves the comparison data. Default: "./results/sample_results_gpt4vision_Yu.json"
    - output_html: The file path for the output HTML file. Default: "comparison_results.html"
"""
############################

import os
import glob
import argparse 
import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def generate_comparison_html(comparison_data, output_file="comparison_results.html"):
    """
    Generate an HTML file visualizing comparison results from multiple methods.

    Args:
    - comparison_data (dict): A dictionary where keys are sample names and values are lists of results.
    - output_file (str): The filename for the output HTML file.

    Returns:
    - None

    Example comparison data:
        comparison_data = {
            "Sample 1": {
                "Result 1": "Description of result 1 for Sample 1",
                "Result 2": "Description of result 2 for Sample 1",
                "Result 3": "Description of result 3 for Sample 1"
            },
            "Sample 2": {
                "Result 1": "Description of result 1 for Sample 2",
                "Result 2": "Description of result 2 for Sample 2",
                "Result 3": "Description of result 3 for Sample 2"
            },
            # Add more resources and their respective results as needed
        }
    """
    # Start building HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparison Results</title>
        <style>
            /* CSS styles for the table */
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            /* CSS styles for the image cell */
            .image-cell {
                width: 200px; /* Adjust width as needed */
            }
            .image-cell img {
                width: 100%;
            }
        </style>
    </head>
    <body>
        <h2>Comparison Results</h2>
        <table>
    """

    # Add the heading row to HTML content
    html_content += "<tr>"
    html_content += f"<th>Image</th>"
    html_content += f"<th>Image Path</th>"

    # Extract method name and add to header as well
    one_sample_result = list(comparison_data.values())[0]
    for method_name in one_sample_result.keys():
        # Add resource name in a separate cell
        html_content += f"<th>{method_name}</th>"
    html_content += "</tr>"        


    # Add comparison data to HTML content
    for image_path, results_of_all_methods in comparison_data.items():
        # Start a new row for each image
        html_content += "<tr>"

        # Add image cell
        html_content += f"<td class='image-cell'><img src={image_path}></td>"
        # Add image name in a separate cell
        image_name = image_path.split("/")[-1]
        html_content += f"<td>{image_name}</td>"
        # Add separate cells for each method
        for method, result in results_of_all_methods.items():
            if isinstance(result, dict):
                # show result with better format if result are saved in dict
                show_result = "<br>".join(f"<b>{key}</b>: {value}" for key, value in result.items())
            elif isinstance(result, list):
                # show result with better format if result are saved in list
                show_result = "<ul>"
                for singe_line in result:
                    show_result += f"<li>{singe_line}</li>"
                show_result += "</ul>"
            else:
                show_result = result
            html_content += f"<td>{show_result}</td>"
        
        # Close the row
        html_content += "</tr>"        

    # Close HTML content
    html_content += """
        </table>
    </body>
    </html>
    """

    # import ipdb; ipdb.set_trace()
    # Write HTML content to file
    with open(output_file, "w") as html_file:
        html_file.write(html_content)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="./results/sample_results_gpt4vision_Yu.json")
    parser.add_argument('--result_ind', type=list, default=[0,3], help="select the results you want to show using index")
    parser.add_argument('--output_html', type=str, default="comparison_results.html")
    args = parser.parse_args()

    # Load json file 
    json_data = load_json(args.input_file)
    desired_index_order = args.result_ind

    # Preprocess data to match the format of Example Comparison Data
    comparison_data = {}
    for sample in json_data:
        # load data for each sample
        sample_index = sample['image_path']
        comparison_per_sample = {}

        # select results you want to visualize using index
        modified_results_list = [sample['results'][i] for i in desired_index_order]
        for result_per_method in modified_results_list:
            # load data (result) for each method
            method_name = result_per_method['instruction']
            method_result_json = result_per_method['prediction']
            try:
                method_result = json.loads(method_result_json.replace('```json', '').replace('```', '').strip())
            except:
                method_result = method_result_json

            comparison_per_sample[method_name] = method_result

        comparison_data[sample_index] = comparison_per_sample

    # Generate HTML file with comparison data
    generate_comparison_html(comparison_data, args.output_html)

    