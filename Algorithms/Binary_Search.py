def BinarySearch(arr,x):
    n=len(arr)
    low=0
    high=n-1
    while low<=high:
        mid=(low+high)//2
        if arr[mid]==x:
            return mid
        elif arr[mid]<x:
            low=mid+1
        else:
            high=mid-1
    return -1
    
arr = [10, 20, 30, 40, 50]
x = 40

result = BinarySearch(arr, x)

if result != -1:
    print("Element found at index:", result)
else:
    print("Element not found")
