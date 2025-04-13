# Project 1: Sorting Algorithms with GUI (Streamlit)

import streamlit as st
import time
import random
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io

# Sorting Algorithms Implementation
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = merge_sort(arr[:mid])
        R = merge_sort(arr[mid:])

        merged = []
        i = j = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                merged.append(L[i])
                i += 1
            else:
                merged.append(R[j])
                j += 1
        merged += L[i:]
        merged += R[j:]
        return merged
    else:
        return arr


def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[i] < arr[l]:
        largest = l

    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        return quick_sort(left) + [pivot] + quick_sort(right)

def quick_sort_3_median(arr):
    if len(arr) <= 1:
        return arr
    first = arr[0]
    mid = arr[len(arr)//2]
    last = arr[-1]
    pivot = sorted([first, mid, last])[1]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort_3_median(left) + middle + quick_sort_3_median(right)

def benchmark_sort(func, arr, runs=3):
    runtimes = []
    for _ in range(runs):
        sample = arr[:]
        gc.collect()
        start = time.perf_counter()
        func(sample)
        end = time.perf_counter()
        runtimes.append((end - start) * 1000)
    return sum(runtimes) / len(runtimes)

# Streamlit UI
st.title("Sorting Algorithms Runtime Comparison")

# Worst-case toggle
case_type = st.radio("Choose input case:", ["Random", "Sorted", "Reversed"], horizontal=True)

# Sorting algorithm map
algorithms = {
    "Bubble Sort": bubble_sort,
    "Insertion Sort": insertion_sort,
    "Selection Sort": selection_sort,
    "Merge Sort": merge_sort,
    "Heap Sort": heap_sort,
    "Quick Sort": quick_sort,
    "Quick Sort (3 Median)": quick_sort_3_median
}

algo_name = st.selectbox("Select a sorting algorithm:", list(algorithms.keys()))
array_size = st.slider("Select the size of the array:", min_value=10, max_value=10000, value=100, step=10)

# Generate input array based on case type
if case_type == "Random":
    arr = [random.randint(1, 100000) for _ in range(array_size)]
elif case_type == "Sorted":
    arr = sorted([random.randint(1, 100000) for _ in range(array_size)])
else:
    arr = sorted([random.randint(1, 100000) for _ in range(array_size)], reverse=True)

st.subheader("Input Array")
st.write(arr)

if st.button("Run Selected Sort"):
    arr_copy = arr[:]
    start = time.perf_counter()
    result = algorithms[algo_name](arr_copy)
    end = time.perf_counter()
    st.subheader("Sorted Output Array")
    st.write(result)
    st.success(f"Runtime for {algo_name}: {(end - start) * 1000:.2f} ms")

# Comparison table + plot
st.subheader("Compare All Sorting Algorithms on the Same Input")
if st.button("Compare All Runtimes"):
    runtimes = {}
    for name, func in algorithms.items():
        avg_time = benchmark_sort(func, arr, runs=3)
        runtimes[name] = avg_time
    df = pd.DataFrame(runtimes.items(), columns=["Algorithm", "Runtime (ms)"]).sort_values(by="Runtime (ms)", ascending=True)
    st.dataframe(df)

    # Horizontal Bar Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = ax2.barh(df["Algorithm"], df["Runtime (ms)"], color='lightcoral')
    for bar in bars:
        width = bar.get_width()
        ax2.annotate(f'{width:.2f} ms',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center')
    ax2.set_xlabel("Runtime (ms)")
    ax2.set_title(f"Sorting Algorithms Runtime Comparison (Array Size = {array_size})")
    ax2.invert_yaxis()
    st.pyplot(fig2)


    buf = io.BytesIO()
    fig2.savefig(buf, format="png")
    st.download_button(
        label="Download Bar Chart as PNG",
        data=buf.getvalue(),
        file_name="sorting_bar_chart.png",
        mime="image/png"
    )


# Interactive Plotly Line Graph with Spinner
st.subheader("Performance of Algorithms across Input Sizes")
if st.button("Generate Interactive Line Graph"):
    input_sizes = list(range(1000, 10001, 1000))
    performance_data = {name: [] for name in algorithms}
    progress = st.progress(0)

    for i, size in enumerate(input_sizes):
        progress.progress((i + 1) / len(input_sizes))
        base_array = [random.randint(1, 100000) for _ in range(size)]

        for name, func in algorithms.items():
            if name in ["Bubble Sort", "Insertion Sort", "Selection Sort"] and size > 3000:
                performance_data[name].append(None)
                continue
            avg_runtime = benchmark_sort(func, base_array, runs=3)
            performance_data[name].append(avg_runtime)

    fig = go.Figure()
    for name, times in performance_data.items():
        fig.add_trace(go.Scatter(x=input_sizes, y=times, mode='lines+markers', name=name, connectgaps=True))

    fig.update_layout(
        title="Sorting Algorithms Runtime Comparison",
        xaxis_title="Input Size",
        yaxis_title="Average Runtime (ms)",
        legend_title="Algorithms",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    plot_buf = io.StringIO()
    fig.write_html(plot_buf, include_plotlyjs='cdn')
    st.download_button(
        label="Download Interactive Chart (HTML)",
        data=plot_buf.getvalue(),
        file_name="interactive_runtime_chart.html",
        mime="text/html"
    )


# Time Complexity Reference Table
st.markdown("""
### Time Complexity Reference
| Algorithm        | Best      | Average   | Worst     |
|------------------|-----------|-----------|-----------|
| Bubble Sort      | O(n)      | O(n²)     | O(n²)     |
| Insertion Sort   | O(n)      | O(n²)     | O(n²)     |
| Selection Sort   | O(n²)     | O(n²)     | O(n²)     |
| Merge Sort       | O(n log n)| O(n log n)| O(n log n)|
| Heap Sort        | O(n log n)| O(n log n)| O(n log n)|
| Quick Sort       | O(n log n)| O(n log n)| O(n²)     |
| Quick Sort 3 Med | O(n log n)| O(n log n)| O(n²) (very rare)    |
""")