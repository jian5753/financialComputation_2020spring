import HW2

def IvEuBsBisection(targetPrice, S0, K, T, r, q, maxError, call = True):
    headIV = 0
    tailIV = 1
    middleIV = (headIV + tailIV) / 2
    if call:
        middlePrice = HW2.vanillaEuro_bs(S0, r, q, middleIV, T, K).callPrice
    else:
        middlePrice = HW2.vanillaEuro_bs(S0, r, q, middleIV, T, K).putPrice

    while abs(middlePrice - targetPrice) > maxError:
        print(middlePrice)
        if middlePrice > targetPrice:
            # 高估 IV
            tailIV = middleIV
        elif middlePrice == targetPrice:
            return middleIV
        elif middlePrice < targetPrice:
            headIV = middleIV

        middleIV = (headIV + tailIV) / 2
        if call:
            middlePrice = HW2.vanillaEuro_bs(S0, r, q, middleIV, T, K).callPrice
        else:
            middlePrice = HW2.vanillaEuro_bs(S0, r, q, middleIV, T, K).putPrice

    return middleIV

def IvBiTreeBisection(targetPrice, S0, K, T, r, q, n, maxError, call = True, Amrc = False):
    headIV = 0
    tailIV = 1
    middleIV = (headIV + tailIV) / 2
    if call:
        if Amrc:  
            entity = HW2.vanillaAmrcCall_svBiTree(S0, r, q, middleIV, T, K, n)
        else:
            entity = HW2.vanillaEuroCall_svBiTree(S0, r, q, middleIV, T, K, n)
    else:
        if Amrc:
            entity = HW2.vanillaAmrcPut_svBiTree(S0, r, q, middleIV, T, K, n)
        else:
            entity = HW2.vanillaEuroPut_svBiTree(S0, r, q, middleIV, T, K, n)

    entity.fillInPayoff()
    middlePrice = entity.payoffTree[0]

    while abs(middlePrice - targetPrice) > maxError:
        print(middlePrice)
        if middlePrice > targetPrice:
            # 高估 IV
            tailIV = middleIV
        elif middlePrice == targetPrice:
            return middleIV
        elif middlePrice < targetPrice:
            headIV = middleIV

        middleIV = (headIV + tailIV) / 2
        if call:
            if Amrc:
                entity = HW2.vanillaAmrcCall_svBiTree(S0, r, q, middleIV, T, K, n)
            else:
                entity = HW2.vanillaEuroCall_svBiTree(S0, r, q, middleIV, T, K, n)
        else:
            if Amrc:
                entity = HW2.vanillaAmrcPut_svBiTree(S0, r, q, middleIV, T, K, n)
            else:
                entity = HW2.vanillaEuroPut_svBiTree(S0, r, q, middleIV, T, K, n)
        entity.fillInPayoff()
        middlePrice = entity.payoffTree[0]

    return middleIV

# def F_forNewton(target, estimatedPrice):
#     return target - estimatedPrice

# EPSILON = 1e-4
        
# def IvEuBsNewton(targetPrice, S0, K, T, r, q, maxError, call = True):
#     IV_i = IvEuBsBisection(targetPrice, S0, K, T, r, q, 0.1, call)
#     if call:
#         priceCap_i = HW2.vanillaEuro_bs(S0, r, q, IV_i, T, K).callPrice
#     else:
#         priceCap_i = HW2.vanillaEuro_bs(S0, r, q, IV_i, T, K).putPrice

#     F = F_forNewton(targetPrice, priceCap_i)

#     while abs(F) > maxError:
#         print(middlePrice, F)
#         if F > 0:
#             # 高估 IV
#             tempIV = IV_i - EPSILON
#             IV_i = IV_i - EPSILON * F(targetPrice, )
#         elif middlePrice == targetPrice:
#             return middleIV
#         elif middlePrice < targetPrice:
#             headIV = middleIV

#         middleIV = (headIV + tailIV) / 2
#         if call:
#             middlePrice = HW2.vanillaEuro_bs(S0, r, q, middleIV, T, K).callPrice
#         else:
#             middlePrice = HW2.vanillaEuro_bs(S0, r, q, middleIV, T, K).putPrice

#     return middleIV