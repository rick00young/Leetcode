import queue
class Solution215():
    # 排序 ：时间复杂度 O(NlogN)，空间复杂度 O(1)
    def findKthLargest_with_sort(self, nums, k):
        new_nums = sorted(nums)
        return new_nums[len(nums)-k]

    # 时间复杂度 O(NlogK)，空间复杂度 O(K)。
    def findKthLarget_with_head(self, nums, k):
        pass

    # 时间复杂度 O(N)，空间复杂度 O(1)
    def findKthLargest_with_quick_sort(self, nums, k):
        pass

    def partition(self, a, l, h):
        i = l
        j = h + 1
        while 1:

        pass

    def swap(self, a, i, j):
        t = a[i]
        a[i] = a[j]
        a[j] = t

###
class Solution347():
    def topKFrequent(self, nums, k):
        frequencyForNum = {}
        for num in nums:
            frequencyForNum[num] = frequencyForNum.get(num, 0) + 1
        buckets = [[] for _ in range(len(nums))]
        for key in frequencyForNum.keys():
            frequency = frequencyForNum.get(key)
            buckets[frequency].append(key)
        # print(frequencyForNum)
        # print(buckets)
        topk = []
        l = 0
        for i in list(range(len(buckets)-1, 0, -1)):
            if l == k:
                break
            if buckets[i]:
                topk.extend(buckets[i])
                l += 1
        return topk

so = Solution347()
nums = [1,1,1,2,2,3]
k = 2
so.topKFrequent(nums, k)

