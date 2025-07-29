#!/bin/bash
set -e

# Example usage:
# bash run.sh -i WebInstruct-CFT-50K -o ans_WebInstruct-CFT-50K -t temp_output_dir_stud/ -s 0 -e 50000

# Function to display usage information
usage() {
  echo "Usage: $0 -i input_path -o output_path -t temp_output_dir -s start_idx -e end_idx"
  exit 1
}

# Parse command-line options
while getopts ":i:o:t:s:e:" opt; do
  case ${opt} in
    i )
      input_path=$OPTARG
      ;;
    o )
      output_path=$OPTARG
      ;;
    t )
      temp_output_dir=$OPTARG
      ;;
    s )
      start_idx=$OPTARG
      ;;
    e )
      end_idx=$OPTARG
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    : )
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done
shift $((OPTIND -1))

# Ensure all required options are provided
if [ -z "$input_path" ] || [ -z "$output_path" ] || [ -z "$temp_output_dir" ] || [ -z "$start_idx" ] || [ -z "$end_idx" ]; then
  echo "All options -i, -o, -t, -s, and -e are required."
  usage
fi

# # Ensure at least one model path is provided
# if [ "$#" -eq 0 ]; then
#   echo "At least one model_path is required."
#   usage
# fi

# Execute the Python script with the provided arguments
python generate_qwen_responses.py \
--model_name "llama3p3_70b" \
--num_processes 64 \
--input_path "$input_path" \
--output_path "$output_path" \
--temp_output_dir "$temp_output_dir" \
--start_idx "$start_idx" \
--end_idx "$end_idx"" and the main code "import json