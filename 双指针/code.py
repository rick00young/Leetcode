import math


class Solution633:
    def judgeSquareSum(self, target):
        print('target:', target)
        if target < 0:
            return False
        i = 0
        j = int(math.sqrt(target))
        while i <= j:
            powSum = i*i + j+j
            if powSum == target:
                return True
            elif powSum > target:
                j -= 1
            else:
                i += 1
        return False


so = Solution633()
so.judgeSquareSum(5)


class Solution345:
    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

    def reverseVowels(self, s):
        if not s:
            return None
        i = 0
        j = len(s)-1
        result = ['' for _ in s]
        s = str(s)
        while i <= j:
            ci = s[i]
            cj = s[j]
            if ci not in self.vowels:
                result[i] = ci
                i += 1
            elif cj not in self.vowels:
                result[j] = cj
                j -= 1
            else:
                result[i] = cj
                result[j] = ci
                i += 1
                j -= 1
        return ''.join(result)

so = Solution345()
so.reverseVowels("leetcode")


class Solution680:
    def isPalindrome(self, s, i, j):
        while i < j:
            print('isPalindrome:',i, j)
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

    def validPalindrome(self, s):
        i = 0
        j = len(s) - 1
        while i < j:
            print('validPalindrome:',i, j)
            if s[i] != s[j]:
                return self.isPalindrome(s, i, j-1) or self.isPalindrome(s, i+1, j)
            i += 1
            j -= 1
        return True
so = Solution680()
so.validPalindrome("abca")

################################

class Solution88:
    def mergeSortedArray(self, nums1, m, nums2, n):
        index1 = m - 1
        index2 = n - 1
        indexMerge = m+n-1
        while index1 >= 0 or index2 >= 0:
            if index1 < 0:
                nums1[indexMerge] = nums2[index2]
                indexMerge -= 1
                index2 -= 1
            elif index2 < 0:
                nums1[indexMerge] = nums1[index1]
                indexMerge -= 1
                index1 -= 1
            elif nums1[index1] > nums2[index2]:
                nums1[indexMerge] = nums1[index1]
                indexMerge -= 1
                index1 -= 1
            else:
                nums1[indexMerge] = nums2[index2]
                indexMerge -= 1
                index2 -= 1
so = Solution88()
nums1, nums2 = [1,2,3,0,0,0], [2,5,6]
so.mergeSortedArray(nums1, 3, nums2, 3)


######

class Solution141:
    def hasCycle(self, head):
        if not head:
            return False
        l1 = head
        l2 = head.next
        while not l1 and not l2 and not l2.next:
            if l1 == l2:
                return True
            l1 = l1.next
            l2 = l2.next.next
        return False

head = object()
so = Solution141()
so.hasCycle(head)