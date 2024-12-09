
# Define the starting number for the new file names
start_number=60

arg1=$1

# Loop through all files matching the pattern city_*.csv
for file in $arg1/cityscapes/city_*.csv; do
    # Extract the current number from the file name using bash string manipulation
    current_number=${file#city_}    # Remove 'city_'
    current_number=${current_number%.csv} # Remove '.csv'
    
    # Calculate the new number
    new_number=$((current_number + start_number))
    
    # Create the new file name
    new_file="city_${new_number}.csv"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
done
# Loop through all files matching the pattern city_*.csv
for file in $arg1/demoviz/city_*.png; do
    # Extract the current number from the file name using bash string manipulation
    current_number=${file#city_}    # Remove 'city_'
    current_number=${current_number%.png} # Remove '.csv'
    
    # Calculate the new number
    new_number=$((current_number + start_number))
    
    # Create the new file name
    new_file="city_${new_number}.png"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
done
# Loop through all files matching the pattern city_*.csv
for file in $arg1/drone_positions/city_*.csv; do
    # Extract the current number from the file name using bash string manipulation
    current_number=${file#city_}    # Remove 'city_'
    current_number=${current_number%.csv} # Remove '.csv'
    
    # Calculate the new number
    new_number=$((current_number + start_number))
    
    # Create the new file name
    new_file="city_${new_number}.csv"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
done
# Loop through all files matching the pattern city_*.csv
for file in $arg1/exportviz/city_*.png; do
    # Extract the current number from the file name using bash string manipulation
    current_number=${file#city_}    # Remove 'city_'
    current_number=${current_number%.png} # Remove '.csv'
    
    # Calculate the new number
    new_number=$((current_number + start_number))
    
    # Create the new file name
    new_file="city_${new_number}.png"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
done
# Loop through all files matching the pattern city_*.csv
for file in $arg1/transparent/city_*.png; do
    # Extract the current number from the file name using bash string manipulation
    current_number=${file#city_}    # Remove 'city_'
    current_number=${current_number%.png} # Remove '.csv'
    
    # Calculate the new number
    new_number=$((current_number + start_number))
    
    # Create the new file name
    new_file="city_${new_number}.png"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
done
# Loop through all files matching the pattern city_*.csv
for file in $arg1/windflow/city_*.png; do
    # Extract the current number from the file name using bash string manipulation
    current_number=${file#city_}    # Remove 'city_'
    current_number=${current_number%.png} # Remove '.csv'
    
    # Calculate the new number
    new_number=$((current_number + start_number))
    
    # Create the new file name
    new_file="city_${new_number}.png"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
done


# Loop through all files matching the pattern city_*.csv
for file in $arg1/lidar/city_*.png; do
    # Extract the current number from the file name using bash string manipulation
    current_number=${file#city_}    # Remove 'city_'
    current_number=${current_number%_pos0.npy} # Remove '_pos0.npy'
    
    # Calculate the new number
    new_number=$((current_number + start_number))
    
    # Create the new file name
    new_file="city_${new_number}_.npy"
    
    # Rename the file
    mv "$file" "$new_file"
    
    echo "Renamed $file to $new_file"
done


echo "Renaming complete."

