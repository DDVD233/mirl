import json
import re
from collections import defaultdict

def extract_boxed_content(text):
    """Extract content within \\boxed{...}"""
    match = re.search(r'\\boxed\{([^}]*)\}', text)
    if match:
        return match.group(1).strip()
    return None

def age_to_bin(age_str):
    """Convert age to bin: a1 (<25), a2 (26-50), a3 (51-75), a4 (75+)"""
    try:
        # Handle different age formats
        age_str = str(age_str).strip()
        if age_str in ['nan', 'None', '']:
            return 'unknown'
        
        # Extract numeric age
        age = float(age_str)
        
        if age <= 25:
            return 'a1'
        elif age <= 50:
            return 'a2'
        elif age <= 75:
            return 'a3'
        else:
            return 'a4'
    except:
        return 'unknown'

def parse_demographics(demo_str):
    """Parse demographics string and return individual demographic attributes"""
    demographics = []
    
    # Split by comma and process each part
    parts = demo_str.split(',')
    
    for part in parts:
        part = part.strip()
        
        # Skip father/mother information
        if 'father:' in part.lower() or 'mother:' in part.lower():
            continue
        
        # Process age
        if 'age:' in part.lower():
            age_match = re.search(r'age:\s*([0-9.]+)', part, re.IGNORECASE)
            if age_match:
                age_bin = age_to_bin(age_match.group(1))
                demographics.append(f"age:{age_bin}")
        
        # Process gender/sex
        if 'gender:' in part.lower():
            gender_match = re.search(r'gender:\s*(\w+)', part, re.IGNORECASE)
            if gender_match:
                gender = gender_match.group(1).lower()
                demographics.append(f"gender:{gender}")
        elif 'sex:' in part.lower():
            sex_match = re.search(r'sex:\s*(\w+)', part, re.IGNORECASE)
            if sex_match:
                sex = sex_match.group(1).lower()
                demographics.append(f"gender:{sex}")
    
    return demographics

def calculate_accuracies(file_path):
    """Load JSON and calculate accuracies per dataset/demographic/label"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Group data by dataset/demographic/label
    groups = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i in range(len(data['predictions'])):
        prediction_text = data['predictions'][i]
        ground_truth = data['ground_truths'][i]
        demographic_str = data['demographics'][i]
        dataset = data['datasets'][i]
        
        # Extract predicted label from boxed content
        predicted = extract_boxed_content(prediction_text)
        
        if predicted:
            # Parse demographics into individual attributes
            demo_attributes = parse_demographics(demographic_str)
            
            # Create entries for each demographic attribute
            for demo_attr in demo_attributes:
                key = f"{dataset}/{demo_attr}/{ground_truth}"
                
                # Update counts
                groups[key]['total'] += 1
                if predicted == ground_truth:
                    groups[key]['correct'] += 1
    
    # Calculate accuracies
    accuracies = {}
    for key, counts in groups.items():
        if counts['total'] > 0:
            accuracy = counts['correct'] / counts['total']
            accuracies[key] = accuracy
    
    return accuracies

def main():
    # Load and process both files
    print("Processing medgemma_grpo.json...")
    grpo_accuracies = calculate_accuracies('medgemma_grpo.json')
    
    print("Processing medgemma_fairgrpo_nd.json...")
    fairgrpo_accuracies = calculate_accuracies('medgemma_fairgrpo_nd.json')
    
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    
    # Count unique groups
    print(f"GRPO unique groups: {len(grpo_accuracies)}")
    print(f"FairGRPO unique groups: {len(fairgrpo_accuracies)}")
    
    # Find keys present in both
    common_keys = set(grpo_accuracies.keys()) & set(fairgrpo_accuracies.keys())
    
    improvements = []
    degradations = []
    
    for key in common_keys:
        diff = fairgrpo_accuracies[key] - grpo_accuracies[key]
        if diff > 0:
            improvements.append((key, diff))
        elif diff < 0:
            degradations.append((key, diff))
    
    print(f"\nTotal groups compared: {len(common_keys)}")
    print(f"Groups where FairGRPO improved: {len(improvements)}")
    print(f"Groups where FairGRPO degraded: {len(degradations)}")
    print(f"Groups with no change: {len(common_keys) - len(improvements) - len(degradations)}")
    
    # Calculate average accuracies
    avg_grpo = sum(grpo_accuracies.values()) / len(grpo_accuracies) if grpo_accuracies else 0
    avg_fairgrpo = sum(fairgrpo_accuracies.values()) / len(fairgrpo_accuracies) if fairgrpo_accuracies else 0
    
    print(f"\nAverage accuracy across all groups:")
    print(f"  GRPO: {avg_grpo:.4f}")
    print(f"  FairGRPO: {avg_fairgrpo:.4f}")
    print(f"  Difference: {avg_fairgrpo - avg_grpo:+.4f}")
    
    # Show sample accuracies by demographic
    print("\n" + "="*80)
    print("Sample Accuracies by Age Group:")
    print("="*80)
    
    for age_group in ['a1', 'a2', 'a3', 'a4']:
        grpo_age_keys = [k for k in grpo_accuracies.keys() if f'/age:{age_group}/' in k]
        fairgrpo_age_keys = [k for k in fairgrpo_accuracies.keys() if f'/age:{age_group}/' in k]
        
        if grpo_age_keys:
            grpo_age_avg = sum(grpo_accuracies[k] for k in grpo_age_keys) / len(grpo_age_keys)
        else:
            grpo_age_avg = 0
            
        if fairgrpo_age_keys:
            fairgrpo_age_avg = sum(fairgrpo_accuracies[k] for k in fairgrpo_age_keys) / len(fairgrpo_age_keys)
        else:
            fairgrpo_age_avg = 0
            
        age_label = {'a1': '<25', 'a2': '26-50', 'a3': '51-75', 'a4': '75+'}[age_group]
        print(f"Age {age_label}: GRPO={grpo_age_avg:.4f}, FairGRPO={fairgrpo_age_avg:.4f}, Diff={fairgrpo_age_avg-grpo_age_avg:+.4f}")
    
    print("\n" + "="*80)
    print("Sample Accuracies by Gender:")
    print("="*80)
    
    for gender in ['male', 'female']:
        grpo_gender_keys = [k for k in grpo_accuracies.keys() if f'/gender:{gender}/' in k]
        fairgrpo_gender_keys = [k for k in fairgrpo_accuracies.keys() if f'/gender:{gender}/' in k]
        
        if grpo_gender_keys:
            grpo_gender_avg = sum(grpo_accuracies[k] for k in grpo_gender_keys) / len(grpo_gender_keys)
        else:
            grpo_gender_avg = 0
            
        if fairgrpo_gender_keys:
            fairgrpo_gender_avg = sum(fairgrpo_accuracies[k] for k in fairgrpo_gender_keys) / len(fairgrpo_gender_keys)
        else:
            fairgrpo_gender_avg = 0
            
        print(f"Gender {gender.capitalize()}: GRPO={grpo_gender_avg:.4f}, FairGRPO={fairgrpo_gender_avg:.4f}, Diff={fairgrpo_gender_avg-grpo_gender_avg:+.4f}")
    
    if improvements:
        print("\n" + "="*80)
        print("Top 10 improvements (FairGRPO vs GRPO):")
        print("="*80)
        improvements.sort(key=lambda x: x[1], reverse=True)
        for key, diff in improvements[:10]:
            grpo_acc = grpo_accuracies.get(key, 0)
            fairgrpo_acc = fairgrpo_accuracies.get(key, 0)
            print(f"  {key}: {grpo_acc:.4f} → {fairgrpo_acc:.4f} (+{diff:.4f})")
    
    if degradations:
        print("\n" + "="*80)
        print("Top 10 degradations (FairGRPO vs GRPO):")
        print("="*80)
        degradations.sort(key=lambda x: x[1])
        for key, diff in degradations[:10]:
            grpo_acc = grpo_accuracies.get(key, 0)
            fairgrpo_acc = fairgrpo_accuracies.get(key, 0)
            print(f"  {key}: {grpo_acc:.4f} → {fairgrpo_acc:.4f} ({diff:.4f})")
    
    # Save results to JSON for further analysis
    results = {
        'grpo_accuracies': grpo_accuracies,
        'fairgrpo_nd_accuracies': fairgrpo_accuracies,
        'summary': {
            'grpo_avg_accuracy': avg_grpo,
            'fairgrpo_avg_accuracy': avg_fairgrpo,
            'accuracy_difference': avg_fairgrpo - avg_grpo,
            'total_groups': len(common_keys),
            'improved_groups': len(improvements),
            'degraded_groups': len(degradations),
            'unchanged_groups': len(common_keys) - len(improvements) - len(degradations)
        }
    }
    
    with open('accuracy_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed results saved to accuracy_comparison.json")

if __name__ == "__main__":
    main()