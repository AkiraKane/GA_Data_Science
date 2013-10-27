--What customers are from the UK

SELECT * FROM [Customers]
where Country = 'UK'



--What is the name of the customer who has the most orders?
SELECT a.CustomerName, count(b.orderid) FROM Customers a join 
Orders b where a.CustomerID = b.CustomerID
group by a.CustomerName
order by count(b.orderid) desc
LIMIT 1

--What supplier has the highest average product price?

SELECT s.SupplierName, avg(p.price) FROM Products p 
join Suppliers s on p.SupplierID = s.SupplierID
group by s.SupplierName
order by avg(p.price) desc
LIMIT 1


--What category has the most orders?
SELECT c.CategoryName, count(od.OrderID) FROM OrderDetails od
join Products p on od.ProductID = p.ProductID
join Categories c on c.CategoryID = p.CategoryID
group by c.CategoryName
order by count(od.OrderID) desc

--What employee made the most sales (by number of sales)?
SELECT e.FirstName, e.LastName, count(o.orderID) FROM Employees e
join Orders o on e.EmployeeID = o.EmployeeID
join OrderDetails od on o.OrderID = od.OrderID
join Products p on p.ProductID = od.ProductID
group by e.FirstName, e.LastName
order by count(o.orderID) desc
LIMIT 1



--What employee made the most sales (by value of sales)?
SELECT e.FirstName, e.LastName, sum(p.Price*od.Quantity) FROM Employees e
join Orders o on e.EmployeeID = o.EmployeeID
join OrderDetails od on o.OrderID = od.OrderID
join Products p on p.ProductID = od.ProductID
group by e.FirstName, e.LastName
order by sum(p.Price*od.Quantity) desc
LIMIT 1

--What Employees have BS degrees? (Hint: Look at LIKE operator)
SELECT * FROM Employees
where Notes like '%BS%'

--What supplier has the highest average product price assuming they have at least 2 products (Hint: Look at the HAVING operator)
SELECT s.SupplierName, avg(p.Price) FROM Products p 
join Suppliers s on p.SupplierID = s.SupplierID
group by s.SupplierName
HAVING COUNT(p.ProductID) >=2




--Submit these SQL queries as a .sql file to schoology, using SQL comments to have the question referring to each:






