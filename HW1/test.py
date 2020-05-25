#%%
class abstract():
    def __init__(self, a, b):
        self.__a = a
        self.__b = b
    
    def _print(self):
        print(self.__a, self.__b)


#%%
class myClass(abstract):
    def __init__(self, a, b, c):
        self._c = c
        super().__init__(a,b)
    def _print(self):
        self.__a        

# %%
test = myClass(10,2,2)

# %%
test._print()

# %%
test.__a

# %%
