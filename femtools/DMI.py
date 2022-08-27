import numpy as np
import math

def comb(n, m):
    return math.factorial(n) / (math.factorial(m) * math.factorial(n - m))

# For testing
def check(x, C):
    return 0 <= x and x < C


# For testing
def get_mechanism(A, B, C):
    assert (len(A) == len(B))
    M = np.zeros((C, C))
    for (x, y) in zip(A, B):
        if check(x, C) and check(y, C):
            M[math.floor(x)][math.floor(y)] += 1
        else:
            raise ValueError(
                'The values of answers must be integers in [0, C)')
    return M


def dmi2(A1, B1, A2, B2, C):
    def check(x):

        return 0 <= x and x < C

    def GetM(A, B):
        assert (len(A) == len(B))
        M = np.zeros((C, C))
        for (x, y) in zip(A, B):
            if check(x) and check(y):
                M[math.floor(x)][math.floor(y)] += 1
            else:
                raise ValueError(
                    'The values of answers must be integers in [0, C)')
        return M

    M1 = GetM(A1, B1)
    M2 = GetM(A2, B2)

    return np.linalg.det(M1) * np.linalg.det(M2)


def DMI(answers, choice_n):
    if type(answers) == list:
        answers = np.array(answers)
    agent_n, task_n = answers.shape

    # T >= 2C; N > 1;
    if task_n < 2 * choice_n:
        raise ValueError('Insufficient number of tasks.')
    if agent_n <= 1:
        raise ValueError('Too few agents.')

    # T tasks are arbitrarily divided into two disjoint parts T_1 , T_2
    answers = np.transpose(answers)
    np.random.shuffle(answers)
    half = task_n // 2
    T1 = np.transpose(answers[:half])
    T2 = np.transpose(answers[half:])
    # Calculate the payment
    payments = calculate_payments(agent_n, choice_n, T1, T2)
    return np.array(payments)


def calculate_payments(agent_n, choice_n, T1, T2):

    if T1.shape[0] - choice_n < 0 or T2.shape[0] - choice_n < 0:
        raise ValueError('Shape of T* cannot be less than choice_n')

    payments = []
    norm_factor = (agent_n - 1) * (math.factorial(choice_n)**2)
    norm_factor *= comb(T1.shape[0], choice_n) * comb(T2.shape[0], choice_n)
    for i in range(agent_n):
        p = 0
        for j in range(agent_n):
            if i == j: continue
            p += dmi2(T1[i], T1[j], T2[i], T2[j], choice_n)
        p /= norm_factor
        payments.append(p)

    return np.array(payments)


import unittest


class TestSum(unittest.TestCase):
    def test_comb_works(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")
        result = comb(8, 4)
        self.assertEqual(result, 70)

    def test_check_works(self):
      x = 0
      y = 0
      c = 4

      result = check(x, c) and check(y, c)
      expected = True
      self.assertEqual(expected, result)

    # Python code translated
    def test_get_mechanism_works(self):
        a1 = np.zeros((4, 4))
        b1 = np.zeros((4, 4))
        c = 4

        a1 = a1[0]
        b1 = b1[0]

        result = get_mechanism(a1, b1, c)
        expected = np.zeros((4, 4))
        expected[0][0] = 4.0
        # self.assertEqual(expected, result)
        np.testing.assert_array_equal(expected, result)

    def test_dmi_inner_works(self):
      a1 = np.zeros((4, 4))
      b1 = np.zeros((4, 4))
      a2 = np.zeros((4, 4))
      b2 = np.zeros((4, 4))

      a1 = a1[0]
      b1 = b1[0]
      a2 = a2[0]
      b2 = b2[0]
      c = 4

      result = dmi2(a1, b1, a2, b2, c)
      self.assertEqual(0.0, result)

    def test_dmi_inner_works_2(self):
      a1 = np.eye(4)
      b1 = np.eye(4)
      a2 = np.eye(4)
      b2 = np.eye(4)

      a1 = a1[0]
      b1 = b1[0]
      a2 = a2[0]
      b2 = b2[0]
      c = 4

      result = dmi2(a1, b1, a2, b2, c)
      self.assertEqual(0.0, result)

    def test_dmi_inner_works_3(self):
      # a1 = np.eye(4)
      a1 = np.full((3, 4), 7.)
      b1 = np.eye(4)
      a2 = np.zeros((4, 4))
      b2 = np.full((4, 4), 3.)

      a1 = a1[1]
      b1 = b1[0]
      a2 = a2[3]
      b2 = b2[2]
      c = 8

      result = dmi2(a1, b1, a2, b2, c)
      self.assertEqual(0.0, result)

    def test_calculate_dmi_works(self):
      a1 = np.full((4, 4), 1)
      result = DMI(a1, 2)
      expected = [0., 0., 0., 0.]
      np.testing.assert_array_equal(expected, result)

    def test_calculate_dmi_works_2(self):
      a1 = np.eye(4)
      result = DMI(a1, 2)
      expected = [0., 0., 0., 0.]
      np.testing.assert_array_equal(expected, result)

    def test_calculate_dmi_works_3(self):
      a1 = np.full((5, 5), 1)
      result = DMI(a1, 2)
      expected = [0., 0., 0., 0., 0.]

    def test_calculate_payments_works(self):
        agent_n = 3
        choice_n = 1
        half = 2

        answers = np.zeros((4,4))
        answers = np.transpose(answers)
        a1 = np.transpose(answers[:half])
        b1 = np.transpose(answers[half:])

        result = calculate_payments(agent_n, choice_n, a1, b1)
        np.testing.assert_array_equal([0.25, 0.25, 0.25], result)

if __name__ == '__main__':
    unittest.main()
