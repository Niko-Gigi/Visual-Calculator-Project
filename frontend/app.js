import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import LinearAlgebra from './components/LinearAlgebra';
import NumericalMethods from './components/NumericalMethods';

export default function App() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Math Visualizer</h1>
      
      <Tabs defaultValue="linear-algebra">
        <TabsList>
          <TabsTrigger value="linear-algebra">Linear Algebra</TabsTrigger>
          <TabsTrigger value="numerical-methods">Numerical Methods</TabsTrigger>
          <TabsTrigger value="ode-methods">Ordinary Differential Equations</TabsTrigger>
          <TabsTrigger value="vector-calculus">Vector Calculus</TabsTrigger>
          <TabsTrigger value="pde-methods">Partial Differential Equations</TabsTrigger>
        </TabsList>

        <TabsContent value="linear-algebra">
          <LinearAlgebra />
        </TabsContent>

        <TabsContent value="numerical-methods">
          <NumericalMethods />
        </TabsContent>

        <TabsContent value="ode-methods">
          <NumericalMethods />
        </TabsContent>

        <TabsContent value="vector-calculus">
          <NumericalMethods />
        </TabsContent>

        <TabsContent value="pde-methods">
          <NumericalMethods />
        </TabsContent>
      </Tabs>
    </div>
  );
}